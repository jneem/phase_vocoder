use clap::{App, Arg};
use hound::{WavReader, WavWriter};

use phase_vocoder::PhaseVocoder;

fn main() {
    let matches = App::new("WAV file stretcher")
        .arg(
            Arg::with_name("INPUT")
                .help("Path to the input file")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("OUTPUT")
                .help("Name of the output file")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("stretch")
                .long("stretch")
                .takes_value(true)
                .required(true),
        )
        .get_matches();

    let samples: Vec<i16> = WavReader::open(matches.value_of("INPUT").unwrap())
        .unwrap()
        .samples::<i16>()
        .map(|x| x.unwrap())
        .collect();

    dbg!(matches.value_of("stretch"));
    let factor: f32 = matches.value_of("stretch").unwrap().parse().unwrap();

    let mut pvoc = PhaseVocoder::new(factor);
    pvoc.input(&samples[..]);
    let mut output = vec![0; 2048];

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(matches.value_of("OUTPUT").unwrap(), spec).unwrap();
    while pvoc.samples_available() > 0 {
        let len = pvoc.samples_available().min(output.len());
        pvoc.consume_output(&mut output[..len]);
        for &x in &output[..len] {
            writer.write_sample(x).unwrap();
        }
    }
    writer.finalize().unwrap();
}
