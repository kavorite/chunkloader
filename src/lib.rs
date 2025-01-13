use std::sync::{Arc, Mutex};

use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::context::input;
use ffmpeg_next::format::Sample;
use ffmpeg_next::media::Type;
use ffmpeg_next::software::resampling::context::Context;
use ffmpeg_next::util::frame::audio::Audio;
use ndarray;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Debug)]
enum AudioError {
    PacketSend(String),
    ResampleError(String),
    FrameReceive(String),
    FileOpen(String),
    NoAudioStream,
    DecoderCreation(String),
    NotStereo(usize),
    ResamplerCreation(String),
}

impl std::error::Error for AudioError {}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AudioError::PacketSend(msg) => write!(f, "Failed to send packet: {}", msg),
            AudioError::ResampleError(msg) => write!(f, "Failed to resample: {}", msg),
            AudioError::FrameReceive(msg) => write!(f, "Failed to receive frame: {}", msg),
            AudioError::FileOpen(msg) => write!(f, "Failed to open file: {}", msg),
            AudioError::NoAudioStream => write!(f, "No audio stream found"),
            AudioError::DecoderCreation(msg) => write!(f, "Failed to get decoder: {}", msg),
            AudioError::NotStereo(channels) => write!(f, "Expected 2 channels, found {}", channels),
            AudioError::ResamplerCreation(msg) => write!(f, "Failed to create resampler: {}", msg),
        }
    }
}

impl From<AudioError> for PyErr {
    fn from(err: AudioError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

#[pyclass]
struct AudioReader {
    decoder: Arc<Mutex<ffmpeg::decoder::Audio>>,
    input: Arc<Mutex<input::Input>>,
    resampler: Arc<Mutex<Context>>,
    stream_index: usize,
    channels: usize,
    buffer: Arc<Mutex<Vec<f32>>>,
    chunk_size: usize,
    source_sample_rate: u32,
    target_sample_rate: Option<u32>,
    total_samples: usize,
}

#[pymethods]
impl AudioReader {
    #[new]
    #[pyo3(signature = (file_path, target_sample_rate=None, chunk_size=1024))]
    fn new(
        file_path: String,
        target_sample_rate: Option<u32>,
        chunk_size: usize,
    ) -> PyResult<Self> {
        ffmpeg::init()
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize FFmpeg: {}", e)))?;

        let input =
            ffmpeg::format::input(&file_path).map_err(|e| AudioError::FileOpen(e.to_string()))?;

        let audio_stream = input
            .streams()
            .best(Type::Audio)
            .ok_or(AudioError::NoAudioStream)?;

        let stream_index = audio_stream.index();
        let decoder = audio_stream
            .codec()
            .decoder()
            .audio()
            .map_err(|e| AudioError::DecoderCreation(e.to_string()))?;

        let source_sample_rate = decoder.rate() as u32;
        let channels = decoder.channels() as usize;
        let channel_layout = decoder.channel_layout();

        if channels != 2 {
            return Err(AudioError::NotStereo(channels).into());
        }

        // Calculate total duration if available
        let total_samples = {
            let duration = audio_stream.duration();
            let time_base = audio_stream.time_base();
            let total_samples = ((duration as f64 * time_base.numerator() as f64
                / time_base.denominator() as f64)
                * source_sample_rate as f64) as usize;
            total_samples
        };

        // Always create a context, either for resampling or for format conversion
        let target_rate = target_sample_rate.unwrap_or(source_sample_rate);
        let resampler = Context::get(
            decoder.format(),
            decoder.channel_layout(),
            source_sample_rate,
            decoder.format(),
            decoder.channel_layout(),
            target_rate,
        )
        .map_err(|e| AudioError::ResamplerCreation(e.to_string()))?;

        Ok(Self {
            decoder: Arc::new(Mutex::new(decoder)),
            input: Arc::new(Mutex::new(input)),
            resampler: Arc::new(Mutex::new(resampler)),
            stream_index,
            channels,
            buffer: Arc::new(Mutex::new(Vec::new())),
            chunk_size,
            source_sample_rate,
            target_sample_rate,
            total_samples,
        })
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.target_sample_rate.unwrap_or(self.source_sample_rate)
    }

    #[getter]
    fn source_sample_rate(&self) -> u32 {
        self.source_sample_rate
    }

    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.total_samples)
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __next__(slf: Bound<'_, Self>) -> PyResult<Option<Bound<'_, PyArray2<f32>>>> {
        let this = slf.borrow_mut();

        let input = this.input.clone();
        let decoder = this.decoder.clone();
        let resampler = this.resampler.clone();
        let buffer = this.buffer.clone();
        let stream_index = this.stream_index;
        let channels = this.channels;
        let chunk_size = this.chunk_size;

        let array = Python::allow_threads(
            slf.py(),
            move || -> Result<Option<ndarray::Array2<f32>>, PyErr> {
                let mut buffer = buffer.lock().unwrap();

                // Keep reading packets until we have enough samples or reach EOF
                while buffer.len() < chunk_size * channels {
                    let mut input = input.lock().unwrap();
                    match input.packets().next() {
                        Some((stream, packet)) => {
                            if stream.index() != stream_index {
                                continue;
                            }

                            let mut frame = Audio::empty();
                            let mut decoder = decoder.lock().unwrap();
                            decoder
                                .send_packet(&packet)
                                .map_err(|e| AudioError::PacketSend(e.to_string()))?;

                            match decoder.receive_frame(&mut frame) {
                                Ok(_) => {
                                    let mut output_frame = Audio::empty();
                                    let mut resampler = resampler.lock().unwrap();
                                    resampler
                                        .run(&frame, &mut output_frame)
                                        .map_err(|e| AudioError::ResampleError(e.to_string()))?;

                                    // Handle both planar and packed formats
                                    if output_frame.is_planar() {
                                        // For planar (LLLLRRRR), interleave the channels
                                        let samples_per_channel = output_frame.samples();
                                        for i in 0..samples_per_channel {
                                            for c in 0..channels {
                                                buffer.push(output_frame.plane::<f32>(c)[i]);
                                            }
                                        }
                                    } else {
                                        // For packed (LRLRLR), just extend
                                        buffer.extend_from_slice(output_frame.plane::<f32>(0));
                                    }
                                }
                                Err(ffmpeg_next::Error::Other { errno: _ }) => continue,
                                Err(e) => {
                                    return Err(AudioError::FrameReceive(e.to_string()).into())
                                }
                            }
                        }
                        None => {
                            // At EOF - return remaining samples or None if buffer is empty
                            if buffer.is_empty() {
                                return Ok(None);
                            }
                            break;
                        }
                    }
                }

                // Extract up to chunk_size samples from the buffer
                let samples_to_take = chunk_size.min(buffer.len() / channels) * channels;
                if samples_to_take == 0 {
                    return Ok(None);
                }

                let chunk: Vec<f32> = buffer.drain(..samples_to_take).collect();
                let array =
                    ndarray::Array2::from_shape_vec((samples_to_take / channels, channels), chunk)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;

                Ok(Some(array))
            },
        )?;

        Ok(array.map(|arr| arr.into_pyarray(slf.py())))
    }
}

#[pymodule]
fn chunkloader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AudioReader>()?;
    Ok(())
}
