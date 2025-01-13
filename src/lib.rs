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

        unsafe {
            // ffmpeg::ffi::av_log_set_level(ffmpeg::ffi::AV_LOG_TRACE as i32);
        }

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
        println!(
            "Creating resampler:\n\
             - Input format: {:?}\n\
             - Input layout: {:?}\n\
             - Input rate: {}\n\
             - Output format: {:?}\n\
             - Output layout: {:?}\n\
             - Output rate: {}",
            decoder.format(),
            decoder.channel_layout(),
            source_sample_rate,
            decoder.format(),
            decoder.channel_layout(),
            target_rate,
        );

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
        let mut this = slf.borrow_mut();
        let py = slf.py();

        let mut buffer = this.buffer.lock().unwrap();
        let mut input = this.input.lock().unwrap();
        let mut decoder = this.decoder.lock().unwrap();
        let mut resampler = this.resampler.lock().unwrap();

        // Keep reading packets until we have enough samples or reach EOF
        while buffer.len() < this.chunk_size * this.channels {
            let mut frame = Audio::empty();

            // Try to receive frames from any remaining packets in decoder
            match decoder.receive_frame(&mut frame) {
                Ok(_) => {
                    let mut output_frame = Audio::empty();
                    output_frame.set_rate(this.target_sample_rate.unwrap_or(this.source_sample_rate) as u32);
                    output_frame.set_format(frame.format());
                    output_frame.set_channel_layout(frame.channel_layout());
                    output_frame.set_channels(frame.channels() as u16);
                    let new_samples = (frame.samples() as u64
                        * this.target_sample_rate.unwrap_or(this.source_sample_rate) as u64
                        / frame.rate() as u64) as usize;
                    output_frame.set_samples(new_samples);

                    match resampler.run(&frame, &mut output_frame) {
                        Ok(_) => {
                            if output_frame.is_planar() {
                                let samples_per_channel = output_frame.samples();
                                for i in 0..samples_per_channel {
                                    for c in 0..this.channels {
                                        buffer.push(output_frame.plane::<f32>(c)[i]);
                                    }
                                }
                            } else {
                                buffer.extend_from_slice(output_frame.plane::<f32>(0));
                            }
                            continue;
                        }
                        Err(e) => {
                            println!(
                                "Warning: Resampler failed, attempting reset:\n\
                                 - samples: {}\n\
                                 - channels: {}\n\
                                 - format: {:?}\n\
                                 - rate: {}\n\
                                 - pts: {:?}\n\
                                 Error: {}",
                                frame.samples(),
                                frame.channels(),
                                frame.format(),
                                frame.rate(),
                                frame.pts(),
                                e
                            );

                            // Create new resampler
                            let new_resampler = Context::get(
                                decoder.format(),
                                decoder.channel_layout(),
                                this.source_sample_rate,
                                decoder.format(),
                                decoder.channel_layout(),
                                this.target_sample_rate.unwrap_or(this.source_sample_rate),
                            )
                            .map_err(|e| AudioError::ResamplerCreation(e.to_string()))?;

                            // Replace the old resampler
                            *resampler = new_resampler;

                            // Try again with the new resampler
                            match resampler.run(&frame, &mut output_frame) {
                                Ok(_) => {
                                    // Handle successful resampling after reset
                                    if output_frame.is_planar() {
                                        let samples_per_channel = output_frame.samples();
                                        for i in 0..samples_per_channel {
                                            for c in 0..this.channels {
                                                buffer.push(output_frame.plane::<f32>(c)[i]);
                                            }
                                        }
                                    } else {
                                        buffer.extend_from_slice(output_frame.plane::<f32>(0));
                                    }
                                }
                                Err(e) => {
                                    println!(
                                        "Warning: Frame still failed after resampler reset: {}",
                                        e
                                    );
                                }
                            }
                            continue;
                        }
                    }
                }
                Err(ffmpeg_next::Error::Other { errno: _ }) => {}
                Err(e) => return Err(AudioError::FrameReceive(e.to_string()).into()),
            }

            // Get next packet
            match input.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() != this.stream_index {
                        continue;
                    }
                    decoder
                        .send_packet(&packet)
                        .map_err(|e| AudioError::PacketSend(e.to_string()))?;
                }
                None => {
                    decoder
                        .send_eof()
                        .map_err(|e| AudioError::PacketSend(e.to_string()))?;
                    if buffer.is_empty() {
                        return Ok(None);
                    }
                    break;
                }
            }
        }

        // Extract up to chunk_size samples from the buffer
        let samples_to_take = this.chunk_size.min(buffer.len() / this.channels) * this.channels;
        if samples_to_take == 0 {
            return Ok(None);
        }

        let chunk: Vec<f32> = buffer.drain(..samples_to_take).collect();
        let array = ndarray::Array2::from_shape_vec(
            (samples_to_take / this.channels, this.channels),
            chunk,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Some(array.into_pyarray(py)))
    }
}

#[pymodule]
fn chunkloader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AudioReader>()?;
    Ok(())
}
