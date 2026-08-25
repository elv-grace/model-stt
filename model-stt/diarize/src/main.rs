//! Speaker diarization for model-stt: raw PCM on stdin, JSON on stdout.
//!
//! speakrs is a Rust library with no binary and no Python bindings, so this is
//! the process boundary between it and the tagger. A subprocess rather than a
//! PyO3 extension because the tagger already shells out to ffmpeg for audio, the
//! call is once per file rather than per frame, and a crash here cannot take the
//! decoder down with it.
//!
//! Audio arrives as raw 16 kHz mono signed 16-bit little-endian samples -- what
//! `ffmpeg -f s16le` writes -- and never as a WAV file. ffmpeg writing a WAV to a
//! pipe leaves placeholder sizes in the header, and a temp file for something
//! already in memory is a cost with no return.
//!
//! Output is one JSON object:
//!
//! ```json
//! {
//!   "mode": "cuda",
//!   "duration": 512.3,
//!   "speakers": ["SPEAKER_00", "SPEAKER_01"],
//!   "segments": [{"start": 0.5, "end": 3.2, "speaker": "SPEAKER_00"}],
//!   "turns":    [{"start": 0.5, "end": 3.4, "speaker": "SPEAKER_00"}],
//!   "embeddings": {"SPEAKER_00": [0.013, ...]}
//! }
//! ```
//!
//! `segments` is exclusive -- one speaker at a time -- so a transcript word maps
//! to exactly one label. `turns` is the same timeline before that collapse, where
//! speakers may overlap; the difference between the two is where a label is
//! least trustworthy, and the caller is expected to say so rather than hide it.
//!
//! The `embeddings` are L2-normalised cluster centroids, and they are what lets
//! the caller keep labels stable across the parts of one asset -- speakrs
//! clusters each run on its own, so SPEAKER_00 in one part has no relation to
//! SPEAKER_00 in the next.
//!
//! This binary decides nothing about the transcript. It reports who spoke when;
//! whether a word gets a label, and what happens to a word that gets none, is
//! the caller's business.

use std::collections::BTreeMap;
use std::error::Error;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use speakrs::pipeline::DiarizationResult;
use speakrs::segment::{merge_segments, Segment};
use speakrs::{AhcConfig, ExecutionMode, PipelineBuilder, PipelineConfig};

type Fallible<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// The only rate the segmentation and embedding models accept.
const SAMPLE_RATE: u32 = 16_000;

const USAGE: &str = "\
Usage: stt-diarize [options] < audio.s16le

Reads raw 16kHz mono s16le PCM on stdin, writes one JSON object on stdout.

Options:
  --mode <cuda|cpu>            execution mode (default: cuda)
  --models-dir <dir>           local model bundle; omit to download from HuggingFace
  --ort-lib <path>             libonnxruntime.so (else $ORT_DYLIB_PATH)
  --merge-gap <seconds>        bridge same-speaker turns closer than this (default: 0)
  --min-duration <seconds>     drop turns shorter than this (default: 0)
  --ahc-threshold <distance>   clustering merge distance; lower splits more (default: 0.6)
  --no-embeddings              omit speaker centroids from the output
  --help
";

struct Args {
    mode: ExecutionMode,
    models_dir: Option<PathBuf>,
    ort_lib: Option<PathBuf>,
    merge_gap: f64,
    min_duration: f64,
    ahc_threshold: Option<f32>,
    embeddings: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("stt-diarize: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Fallible<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        print!("{USAGE}");
        return Ok(());
    };

    // Set before anything touches ort: speakrs resolves the runtime once, on
    // first use, and caches the result for the process.
    if let Some(path) = &args.ort_lib {
        std::env::set_var("ORT_DYLIB_PATH", path);
    }

    // Models load before the audio is read, not after. Loading is what fails
    // when the ONNX Runtime or the GPU is wrong, and failing then costs the
    // caller a pipe error instead of a whole file decoded for nothing.
    let mut builder = match &args.models_dir {
        Some(dir) => PipelineBuilder::from_dir(dir, args.mode),
        None => PipelineBuilder::from_pretrained(args.mode)?,
    };
    if let Some(threshold) = args.ahc_threshold {
        // for_mode first, so overriding the clustering threshold does not also
        // discard the mode's own defaults
        let mut config = PipelineConfig::for_mode(args.mode);
        config.ahc = AhcConfig { threshold };
        builder = builder.pipeline(config);
    }
    let mut pipeline = builder.build()?;

    let audio = read_pcm(io::stdin().lock())?;
    if audio.is_empty() {
        return Err("no audio on stdin".into());
    }
    let duration = audio.len() as f64 / f64::from(SAMPLE_RATE);

    let result = pipeline.run(&audio)?;

    let segments = exclusive_segments(&result, args.merge_gap, args.min_duration);
    let turns = ordered(result.segments.clone(), args.min_duration);
    let embeddings = if args.embeddings {
        speaker_centroids(&result)
    } else {
        BTreeMap::new()
    };

    // Every label that appears in the timeline, whether or not a centroid could
    // be computed for it.
    let mut speakers: Vec<&str> = segments.iter().map(|s| s.speaker.as_str()).collect();
    speakers.sort_unstable();
    speakers.dedup();

    let output = serde_json::json!({
        "mode": mode_name(args.mode),
        "sample_rate": SAMPLE_RATE,
        "duration": round3(duration),
        "speakers": speakers,
        "segments": as_json(&segments),
        "turns": as_json(&turns),
        "embeddings": embeddings,
    });

    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &output)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

/// One speaker at a time, ordered by time.
///
/// `make_exclusive` keeps only the highest-scoring speaker in each frame, which
/// is what makes a word's label a single answer rather than a set. Overlapped
/// speech is not lost so much as decided: the stronger speaker wins the frame.
/// `turns` in the output is the same timeline with that decision not yet made.
///
/// Merging runs before sorting because `to_segments` emits one speaker's whole
/// timeline at a time, and `merge_segments` only joins neighbours in the list.
fn exclusive_segments(
    result: &DiarizationResult,
    merge_gap: f64,
    min_duration: f64,
) -> Vec<Segment> {
    let mut exclusive = result.discrete_diarization.clone();
    exclusive.make_exclusive();

    let mut segments = exclusive.to_segments();
    if merge_gap > 0.0 {
        segments = merge_segments(&segments, merge_gap);
    }
    ordered(segments, min_duration)
}

fn ordered(mut segments: Vec<Segment>, min_duration: f64) -> Vec<Segment> {
    if min_duration > 0.0 {
        segments.retain(|s| s.duration() >= min_duration);
    }
    segments.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.speaker.cmp(&b.speaker))
    });
    segments
}

fn as_json(segments: &[Segment]) -> Vec<serde_json::Value> {
    segments
        .iter()
        .map(|s| {
            serde_json::json!({
                "start": round3(s.start),
                "end": round3(s.end),
                "speaker": s.speaker,
            })
        })
        .collect()
}

/// Mean embedding per speaker, L2-normalised, keyed by the same label the
/// segments carry.
///
/// speakrs cuts the audio into 10s windows and gives each window up to three
/// local speaker slots, at most two of them talking at once -- the shape of
/// pyannote's segmentation-3.0 powerset, fixed by the weights. `hard_clusters`
/// maps (window, slot) to the recording-wide cluster that slot was assigned to,
/// and a cluster id indexes a column of the reconstructed activations, where
/// `segment::to_segments` names column *n* `SPEAKER_nn`. So the id is the label,
/// and the three-slot limit bounds voices per window, never per recording.
///
/// A negative id is a slot with nothing in it: -2 for inactive in that window,
/// -1 for never assigned to any cluster. Their embeddings come from an empty
/// mask and are non-finite, so averaging them in would corrupt a centroid.
/// Skipping them changes the centroids only. It removes no speech and no
/// segment: the timeline is built from the reconstructed activations, not from
/// this table.
fn speaker_centroids(result: &DiarizationResult) -> BTreeMap<String, Vec<f32>> {
    let (windows, slots, dim) = result.embeddings.dim();
    let (cluster_windows, cluster_slots) = result.hard_clusters.dim();

    let mut sums: BTreeMap<i32, (Vec<f32>, usize)> = BTreeMap::new();
    for window in 0..windows.min(cluster_windows) {
        for slot in 0..slots.min(cluster_slots) {
            let cluster = result.hard_clusters[[window, slot]];
            if cluster < 0 {
                continue;
            }
            let values: Vec<f32> = (0..dim)
                .map(|d| result.embeddings[[window, slot, d]])
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let entry = sums.entry(cluster).or_insert_with(|| (vec![0.0; dim], 0));
            for (acc, value) in entry.0.iter_mut().zip(&values) {
                *acc += value;
            }
            entry.1 += 1;
        }
    }

    sums.into_iter()
        .filter_map(|(cluster, (sum, count))| {
            // normalising the sum and normalising the mean give the same vector
            let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt();
            if count == 0 || !norm.is_normal() {
                return None;
            }
            let centroid = sum.iter().map(|v| v / norm).collect();
            Some((format!("SPEAKER_{cluster:02}"), centroid))
        })
        .collect()
}

fn read_pcm(mut reader: impl Read) -> io::Result<Vec<f32>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    // a truncated trailing byte is not a sample
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0)
        .collect())
}

fn mode_name(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Cuda => "cuda",
        ExecutionMode::CudaFast => "cuda-fast",
        _ => "cpu",
    }
}

/// Milliseconds are the tagger's unit; three decimals of seconds is exact there
/// and keeps the JSON readable.
fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

impl Args {
    /// `Ok(None)` means --help was asked for.
    fn parse(args: impl Iterator<Item = String>) -> Fallible<Option<Self>> {
        let mut parsed = Args {
            mode: ExecutionMode::Cuda,
            models_dir: None,
            ort_lib: None,
            merge_gap: 0.0,
            min_duration: 0.0,
            ahc_threshold: None,
            embeddings: true,
        };

        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let mut value = || {
                args.next()
                    .ok_or_else(|| format!("{flag} needs a value\n\n{USAGE}"))
            };
            match flag.as_str() {
                "--help" | "-h" => return Ok(None),
                "--mode" => {
                    let name = value()?;
                    parsed.mode = match name.as_str() {
                        "cpu" => ExecutionMode::Cpu,
                        "cuda" => ExecutionMode::Cuda,
                        // DISABLED (cuda-fast): 2s window step instead of 1s, which
                        // puts speaker changes further from the word they happened on.
                        // "cuda-fast" => ExecutionMode::CudaFast,
                        other => return Err(format!("unknown mode {other:?}\n\n{USAGE}").into()),
                    };
                }
                "--models-dir" => parsed.models_dir = Some(PathBuf::from(value()?)),
                "--ort-lib" => parsed.ort_lib = Some(PathBuf::from(value()?)),
                "--merge-gap" => parsed.merge_gap = value()?.parse()?,
                "--min-duration" => parsed.min_duration = value()?.parse()?,
                "--ahc-threshold" => parsed.ahc_threshold = Some(value()?.parse()?),
                "--no-embeddings" => parsed.embeddings = false,
                other => return Err(format!("unknown option {other:?}\n\n{USAGE}").into()),
            }
        }

        Ok(Some(parsed))
    }
}
