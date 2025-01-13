use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::context::input;
use ffmpeg_next::format::input;
use ffmpeg_next::media::Type;
use ffmpeg_next::software::resampler;
use ffmpeg_next::software::resampling::context::Context;
use ffmpeg_next::util::frame::audio::Audio;
use ndarray;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::BoundObject;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("FFmpeg system initialization failed: {0}")]
    SystemInit(String),

    #[error("FFmpeg configuration error: {0}")]
    Config(String),

    #[error("Failed to initialize FFmpeg: {0}")]
    FFmpegInit(String),

    #[error("Failed to open audio file: {0}")]
    FileOpen(String),

    #[error("Failed to create decoder: {0}")]
    DecoderCreation(String),

    #[error("Failed to send packet to decoder: {0}")]
    PacketSend(String),

    #[error("Failed to receive frame from decoder: {0}")]
    FrameReceive(String),

    #[error("Failed to create resampler: {0}")]
    ResamplerCreation(String),

    #[error("Failed to resample audio: {0}")]
    ResampleError(String),

    #[error("Not a stereo audio file (got {0} channels)")]
    NotStereo(usize),

    #[error("Audio stream not initialized")]
    NotInitialized,

    #[error("No audio stream found in file")]
    NoAudioStream,

    #[error("Internal state error: {0}")]
    InternalState(String),
}

impl From<ffmpeg_next::Error> for AudioError {
    fn from(err: ffmpeg_next::Error) -> Self {
        match err {
            ffmpeg_next::Error::Bug => {
                AudioError::InternalState(format!("Internal FFmpeg error: {}", err))
            }
            ffmpeg_next::Error::Bug2 => {
                AudioError::InternalState(format!("Internal FFmpeg error 2: {}", err))
            }
            ffmpeg_next::Error::Exit => {
                AudioError::InternalState(format!("FFmpeg process exit: {}", err))
            }
            ffmpeg_next::Error::External => {
                AudioError::InternalState(format!("External FFmpeg error: {}", err))
            }
            ffmpeg_next::Error::InvalidData => {
                AudioError::Config(format!("Invalid FFmpeg configuration or data: {}", err))
            }
            ffmpeg_next::Error::PatchWelcome => {
                AudioError::InternalState(format!("FFmpeg patch welcome: {}", err))
            }
            ffmpeg_next::Error::DecoderNotFound => {
                AudioError::DecoderCreation(format!("Required decoder not found: {}", err))
            }
            ffmpeg_next::Error::EncoderNotFound => {
                AudioError::Config(format!("Required encoder not found: {}", err))
            }
            ffmpeg_next::Error::StreamNotFound => AudioError::NoAudioStream,
            ffmpeg_next::Error::Unknown => {
                AudioError::InternalState(format!("Unknown FFmpeg error: {}", err))
            }
            ffmpeg_next::Error::BufferTooSmall => {
                AudioError::Config(format!("Buffer too small: {}", err))
            }
            ffmpeg_next::Error::Experimental => {
                AudioError::Config(format!("Experimental feature: {}", err))
            }
            ffmpeg_next::Error::InputChanged => {
                AudioError::Config(format!("Input format changed: {}", err))
            }
            ffmpeg_next::Error::OutputChanged => {
                AudioError::Config(format!("Output format changed: {}", err))
            }
            ffmpeg_next::Error::FilterNotFound => {
                AudioError::Config(format!("Required filter not found: {}", err))
            }
            ffmpeg_next::Error::OptionNotFound => {
                AudioError::Config(format!("Required option not found: {}", err))
            }
            ffmpeg_next::Error::Eof => AudioError::InternalState("End of file reached".to_string()),
            ffmpeg_next::Error::BsfNotFound => {
                AudioError::Config(format!("Bitstream filter not found: {}", err))
            }
            // Add any other variants that might be missing
            _ => AudioError::InternalState(format!("Unhandled FFmpeg error: {}", err)),
        }
    }
}

impl From<AudioError> for PyErr {
    fn from(err: AudioError) -> PyErr {
        PyValueError::new_err(format!("{}", err))
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
}

#[pymethods]
impl AudioReader {
    #[new]
    fn new(file_path: String, target_sample_rate: u32, chunk_size: usize) -> PyResult<Self> {
        ffmpeg::init().map_err(|e| AudioError::FFmpegInit(e.to_string()))?;

        let input = input(&file_path).map_err(|e| AudioError::FileOpen(e.to_string()))?;

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
        let channels = decoder.channels() as usize;

        if channels != 2 {
            return Err(AudioError::NotStereo(channels).into());
        }

        let resampler = decoder
            .resampler(
                ffmpeg::util::format::Sample::F32(ffmpeg::util::format::sample::Type::Planar),
                ffmpeg::util::channel_layout::ChannelLayout::STEREO,
                target_sample_rate,
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
        })
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __next__(slf: Bound<'_, Self>) -> PyResult<Option<Bound<'_, PyArray2<f32>>>> {
        let this = slf.borrow_mut();

        let input = Arc::clone(&this.input);
        let decoder = Arc::clone(&this.decoder);
        let resampler = Arc::clone(&this.resampler);
        let buffer = Arc::clone(&this.buffer);
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

                                    // Append new samples to our buffer
                                    buffer.extend_from_slice(output_frame.plane::<f32>(0));
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
