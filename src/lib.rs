use rustfft::num_complex::Complex;
use rustfft::FFTplanner;
use std::collections::VecDeque;

const WINDOW_SIZE: usize = 2048;
const HOP_SIZE: usize = WINDOW_SIZE / 8;

struct Synthesis {
    output: VecDeque<f32>,

    // This points at the first *unfinished* sample in the output buffer. Everything up until
    // `pos` is eligible to be consumed using `consume_output`.
    pos: usize,
    hop_size: usize,

    fft: FFTplanner<f32>,
    fft_input: Vec<Complex<f32>>,
    fft_output: Vec<Complex<f32>>,
    hanning: Vec<f32>,
}

impl Synthesis {
    pub fn new(factor: f32) -> Synthesis {
        Synthesis {
            output: VecDeque::new(),
            pos: 0,
            hop_size: ((HOP_SIZE as f32) / factor) as usize,
            fft: FFTplanner::new(true),
            fft_input: vec![0.0.into(); WINDOW_SIZE],
            fft_output: vec![0.0.into(); WINDOW_SIZE],
            hanning: apodize::hanning_iter(WINDOW_SIZE)
                .map(|x| x as f32)
                .collect(),
        }
    }

    pub fn reset(&mut self, factor: f32) {
        self.output.clear();
        self.pos = 0;
        self.hop_size = ((HOP_SIZE as f32) / factor) as usize;
    }

    pub fn advance(&mut self, phase: &[f32], mag: &[f32]) {
        assert!(phase.len() == WINDOW_SIZE / 2);
        assert!(mag.len() == WINDOW_SIZE / 2);

        for i in 0..phase.len() {
            self.fft_input[i] = Complex::from_polar(&mag[i], &phase[i]);
            self.fft_input[WINDOW_SIZE - i - 1] = self.fft_input[i].conj();
        }
        self.fft
            .plan_fft(WINDOW_SIZE)
            .process(&mut self.fft_input[..], &mut self.fft_output[..]);

        self.output.resize(self.pos + WINDOW_SIZE, 0.0);
        // FIXME: I'm not sure why the factor of 4 here, but without it we end up making things really quiet.
        let norm = 4.0 / (WINDOW_SIZE as f32).sqrt();
        for i in 0..WINDOW_SIZE {
            self.output[self.pos + i] += self.hanning[i] * self.fft_output[i].re * norm;
        }

        self.pos += self.hop_size;
    }

    pub fn samples_available(&self) -> usize {
        self.pos
    }

    pub fn consume_output(&mut self, buf: &mut [i16]) {
        assert!(self.samples_available() >= buf.len());
        for x in buf.iter_mut() {
            let sample = self.output.pop_front().unwrap();
            *x = sample.max(std::i16::MIN as f32).min(std::i16::MAX as f32) as i16;
        }
        self.pos -= buf.len();
    }
}

struct Analysis {
    accumulated_phase: Vec<f32>,
    cur_phase: Vec<f32>,
    last_phase: Vec<f32>,
    cur_magnitude: Vec<f32>,
    peaks: Vec<usize>,
    delta_phase: Vec<f32>,
    factor: f32,
}

impl Analysis {
    pub fn new(factor: f32) -> Analysis {
        Analysis {
            accumulated_phase: vec![0.0; WINDOW_SIZE / 2],
            cur_phase: vec![0.0; WINDOW_SIZE / 2],
            last_phase: vec![0.0; WINDOW_SIZE / 2],
            cur_magnitude: vec![0.0; WINDOW_SIZE / 2],
            peaks: (0..(WINDOW_SIZE / 2)).collect(),
            delta_phase: Vec::with_capacity(WINDOW_SIZE / 2),
            factor,
        }
    }

    pub fn reset(&mut self, factor: f32) {
        for x in &mut self.accumulated_phase {
            *x = 0.0;
        }
        for x in &mut self.cur_phase {
            *x = 0.0;
        }
        self.peaks.clear();
        for i in 0..(WINDOW_SIZE / 2) {
            self.peaks.push(i);
        }
        self.factor = factor;
    }

    fn find_peaks(&mut self) {
        // FIXME: what's a good threshold value?
        const THRESHOLD: f32 = 150.0;

        self.peaks.clear();
        let mut i = 2;
        while i < self.cur_magnitude.len() - 2 {
            let window = &self.cur_magnitude[(i - 2)..(i + 3)];
            let max = window.iter().cloned().fold(0.0f32, |x, y| x.max(y));
            if max < THRESHOLD {
                i += 2;
            } else if window[2] == max {
                self.peaks.push(i);
                i += 2;
            } else {
                i += 1;
            }
        }

        if self.peaks.is_empty() {
            self.peaks.push(1);
        }
    }

    pub fn advance(&mut self, fft: &[Complex<f32>]) {
        std::mem::swap(&mut self.cur_phase, &mut self.last_phase);

        assert!(fft.len() == WINDOW_SIZE);
        for i in 0..(WINDOW_SIZE / 2) {
            self.cur_phase[i] = fft[i].arg();
            self.cur_magnitude[i] = fft[i].norm();
        }

        self.delta_phase.clear();
        for &idx in &self.peaks {
            let cur_phase = self.cur_phase[idx];
            let last_phase = self.last_phase[idx];
            let delta = (cur_phase - last_phase) / self.factor;
            self.delta_phase.push(delta);
        }

        // Divide into regions around each peak.
        let mut start_idx = 0;
        for (peak_count, w) in self.peaks.windows(2).enumerate() {
            let (peak_idx, next_peak_idx) = (w[0], w[1]);
            let end_idx = (peak_idx + next_peak_idx) / 2 + 1;
            let delta = self.accumulated_phase[peak_idx] - self.cur_phase[peak_idx]
                + self.delta_phase[peak_count];

            for i in start_idx..end_idx {
                self.accumulated_phase[i] = self.cur_phase[i] + delta;
            }
            start_idx = end_idx;
        }
        if start_idx < self.accumulated_phase.len() {
            let peak_idx = *self.peaks.last().unwrap();
            let delta = self.accumulated_phase[peak_idx] - self.cur_phase[peak_idx]
                + self.delta_phase.last().unwrap();
            for i in start_idx..self.accumulated_phase.len() {
                self.accumulated_phase[i] = self.cur_phase[i] + delta;
            }
        }

        self.find_peaks();
    }

    pub fn phase(&self) -> &[f32] {
        &self.accumulated_phase[..]
    }

    pub fn magnitude(&self) -> &[f32] {
        &self.cur_magnitude[..]
    }
}

// TODO
// - maybe Stfft is a more common name
struct WindowedFft {
    // The input. When we're done with input, we pop it from the front.
    input: VecDeque<f32>,

    // A precomputed Hanning window.
    hanning: Vec<f32>,

    // A buffer for storing the Hanning window multiplied by the input.
    // NOTE: this is complex for now because rustfft currently only supports
    // complex input.
    windowed_input: Vec<Complex<f32>>,

    fft: FFTplanner<f32>,
    out: Vec<Complex<f32>>,
}

impl WindowedFft {
    pub fn new() -> WindowedFft {
        WindowedFft {
            input: VecDeque::new(),
            hanning: apodize::hanning_iter(WINDOW_SIZE)
                .map(|x| x as f32)
                .collect(),
            fft: FFTplanner::new(false),
            out: vec![Complex::new(0.0, 0.0); WINDOW_SIZE],
            windowed_input: vec![Complex::new(0.0, 0.0); WINDOW_SIZE],
        }
    }

    pub fn reset(&mut self) {
        self.input.clear();
    }

    // Multiple the current input window by the Hanning filter and put the result in `self.windowed_input`.
    fn mult_window(&mut self) {
        assert!(self.input.len() >= WINDOW_SIZE);

        for i in 0..WINDOW_SIZE {
            self.windowed_input[i] = (self.input[i] * self.hanning[i]).into();
        }
    }

    // Take the FFT of the data in `self.windowed_input`.
    fn fft(&mut self) {
        let fft = self.fft.plan_fft(WINDOW_SIZE);
        fft.process(&mut self.windowed_input[..], &mut self.out[..]);

        let norm_factor: Complex<f32> = (1.0 / (WINDOW_SIZE as f32).sqrt()).into();
        for x in &mut self.out {
            *x *= norm_factor;
        }
    }

    pub fn advance(&mut self) -> bool {
        if self.input.len() < WINDOW_SIZE {
            return false;
        }

        self.mult_window();
        self.fft();

        self.input.drain(0..HOP_SIZE);
        return true;
    }

    pub fn output(&self) -> &[Complex<f32>] {
        &self.out[..]
    }

    pub fn input<I: IntoIterator<Item = f32>>(&mut self, iter: I) {
        self.input.extend(iter.into_iter());
    }
}

pub struct PhaseVocoder {
    wfft: WindowedFft,
    ana: Analysis,
    syn: Synthesis,
}

impl PhaseVocoder {
    pub fn new(factor: f32) -> PhaseVocoder {
        PhaseVocoder {
            wfft: WindowedFft::new(),
            ana: Analysis::new(factor),
            syn: Synthesis::new(factor),
        }
    }

    pub fn reset(&mut self, factor: f32) {
        self.wfft.reset();
        self.ana.reset(factor);
        self.syn.reset(factor);
    }

    pub fn samples_available(&self) -> usize {
        self.syn.samples_available()
    }

    pub fn consume_output(&mut self, buf: &mut [i16]) {
        self.syn.consume_output(buf)
    }

    pub fn input(&mut self, buf: &[i16]) {
        self.wfft.input(buf.iter().map(|&x| x as f32));

        while self.wfft.advance() {
            self.ana.advance(self.wfft.output());
            self.syn.advance(self.ana.phase(), self.ana.magnitude());
        }
    }
}
