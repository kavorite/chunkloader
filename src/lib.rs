use std::sync::{Arc, Mutex};

use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::context::input;
use ffmpeg_next::media::Type;
use ffmpeg_next::software::resampling::context::Context;
use ffmpeg_next::util::frame::audio::Audio;
use ndarray;
use numpy::{IntoPyArray, PyArray2};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ndarray::s;

#[derive(Debug)]
enum AudioError {
    PacketSend(String),
    FileOpen(String),
    NoAudioStream,
    DecoderCreation(String),
    ResamplerCreation(String),
    FFmpeg(String),
    Conversion(String),
}

impl std::error::Error for AudioError {}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AudioError::PacketSend(msg) => write!(f, "Failed to send packet: {}", msg),
            AudioError::FileOpen(msg) => write!(f, "Failed to open file: {}", msg),
            AudioError::NoAudioStream => write!(f, "No audio stream found"),
            AudioError::DecoderCreation(msg) => write!(f, "Failed to get decoder: {}", msg),
            AudioError::ResamplerCreation(msg) => write!(f, "Failed to create resampler: {}", msg),
            AudioError::FFmpeg(msg) => write!(f, "FFmpeg error: {}", msg),
            AudioError::Conversion(msg) => write!(f, "Conversion error: {}", msg),
        }
    }
}

impl From<AudioError> for PyErr {
    fn from(err: AudioError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

impl From<ffmpeg_next::Error> for AudioError {
    fn from(err: ffmpeg_next::Error) -> Self {
        AudioError::FFmpeg(err.to_string())
    }
}

#[pyclass]
#[derive(Clone)]
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
    force_mono: bool,
}

#[pyclass]
struct AudioReaderIterator {
    reader: AudioReader,
}

#[pymethods]
impl AudioReader {
    #[new]
    #[pyo3(signature = (file_path, target_sample_rate=None, chunk_size=1024, force_mono=false))]
    fn new(
        file_path: String,
        target_sample_rate: Option<u32>,
        chunk_size: usize,
        force_mono: bool,
    ) -> PyResult<Self> {
        ffmpeg::init()
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize FFmpeg: {}", e)))?;

        unsafe {
            ffmpeg::ffi::av_log_set_level(ffmpeg::ffi::AV_LOG_QUIET as i32);
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
        let _channel_layout = decoder.channel_layout();

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
            force_mono,
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
        // First convert to output channels to avoid losing samples in integer division
        let samples_per_channel = self.total_samples / self.channels;

        // Adjust for resampling
        let resampled_samples = if let Some(target_rate) = self.target_sample_rate {
            (samples_per_channel * target_rate as usize).div_ceil(self.source_sample_rate as usize)
        } else {
            samples_per_channel
        };

        // Adjust for mono conversion if enabled
        let output_channels = if self.force_mono { 1 } else { self.channels };

        Ok(resampled_samples * output_channels)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<AudioReaderIterator> {
        Ok(AudioReaderIterator {
            reader: slf.clone(),
        })
    }

    #[getter]
    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn decode_samples(&self, num_samples: usize, py: Python<'_>) -> PyResult<Py<PyArray2<f32>>> {
        let output_channels = if self.force_mono { 1 } else { self.channels };
        let buffer_arc = Arc::clone(&self.buffer);
        let force_mono = self.force_mono;

        // Release GIL for processing
        let result = py.allow_threads(move || {
            let mut collected_samples = Vec::with_capacity(num_samples * output_channels);
            let mut buffer = buffer_arc.lock().unwrap();

            // First, check if we have enough samples in the buffer
            if buffer.len() >= num_samples * output_channels {
                collected_samples.extend(buffer.drain(..num_samples * output_channels));
                return Ok(collected_samples);
            }

            // If not, extend with what we have in buffer
            collected_samples.extend(buffer.drain(..));

            // Keep processing until we have enough samples or hit EOF
            while collected_samples.len() < num_samples * output_channels {
                match process_audio_data(&self, &mut buffer) {
                    Ok(chunk) => {
                        if chunk.is_empty() {
                            break;
                        }
                        let samples_needed =
                            num_samples * output_channels - collected_samples.len();
                        let samples_to_take = samples_needed.min(chunk.len());
                        // Ensure we take a multiple of output_channels
                        let samples_to_take = samples_to_take - (samples_to_take % output_channels);
                        if samples_to_take > 0 {
                            collected_samples.extend(&chunk[..samples_to_take]);
                        }
                    }
                    Err(e) => {
                        if e.to_string().contains("End of file") {
                            break;
                        }
                        return Err(e);
                    }
                }
            }

            // Ensure final length is a multiple of output_channels
            let truncate_to = collected_samples.len() - (collected_samples.len() % output_channels);
            collected_samples.truncate(truncate_to);

            Ok(collected_samples)
        })?;

        let actual_samples = result.len() / output_channels;
        let shape = if force_mono {
            (actual_samples, 1)
        } else {
            (actual_samples, output_channels)
        };

        // Create array with GIL - will be empty (0 samples) if no data was available
        let array = ndarray::Array2::from_shape_vec(shape, result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(array.into_pyarray(py).into())
    }
}

#[pymethods]
impl AudioReaderIterator {
    fn __next__(slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyArray2<f32>>>> {
        let reader = slf.reader.clone();
        let output_channels = if reader.force_mono {
            1
        } else {
            reader.channels
        };
        let buffer_arc = Arc::clone(&reader.buffer);
        let chunk_size = reader.chunk_size;
        let force_mono = reader.force_mono;

        // Clone buffer_arc for the closure
        let buffer_arc_for_thread = Arc::clone(&buffer_arc);

        // Release GIL for all FFmpeg and buffer operations
        let result = py.allow_threads(move || {
            let mut buffer = buffer_arc_for_thread.lock().unwrap();
            process_audio_data(&reader, &mut buffer)
        });

        match result {
            Ok(chunk_data) => {
                if chunk_data.is_empty() {
                    return Ok(None);
                }

                let shape = if force_mono {
                    (chunk_data.len(), 1)
                } else {
                    (chunk_data.len() / output_channels, output_channels)
                };

                // Minimize time holding the GIL - just for array creation
                let array = ndarray::Array2::from_shape_vec(shape, chunk_data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;

                Ok(Some(array.into_pyarray(py).into()))
            }
            Err(e) => {
                if e.to_string().contains("End of file") {
                    // Only treat EOF as normal if we have no data to return
                    let buffer = buffer_arc.lock().unwrap();
                    if buffer.is_empty() {
                        Ok(None)
                    } else {
                        drop(buffer);
                        // Process the remaining buffer data
                        let mut buffer = buffer_arc.lock().unwrap();
                        let samples_to_take =
                            chunk_size.min(buffer.len() / output_channels) * output_channels;
                        let chunk: Vec<f32> = buffer.drain(..samples_to_take).collect();
                        let array = if force_mono {
                            ndarray::Array2::from_shape_vec((samples_to_take, 1), chunk)
                        } else {
                            ndarray::Array2::from_shape_vec(
                                (samples_to_take / output_channels, output_channels),
                                chunk,
                            )
                        }
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;

                        Ok(Some(array.into_pyarray(py).into()))
                    }
                } else {
                    Err(e.into())
                }
            }
        }
    }
}

// Move the audio processing logic to a separate function to keep code organized
fn process_audio_data(reader: &AudioReader, buffer: &mut Vec<f32>) -> Result<Vec<f32>, AudioError> {
    let mut input = reader.input.lock().unwrap();
    let mut decoder = reader.decoder.lock().unwrap();
    let mut resampler = reader.resampler.lock().unwrap();
    let output_channels = if reader.force_mono {
        1
    } else {
        reader.channels
    };

    // Process packets until we have enough samples or reach EOF
    while buffer.len() < reader.chunk_size * reader.channels {
        let mut frame = Audio::empty();

        // Try to receive frames from any remaining packets in decoder
        match decoder.receive_frame(&mut frame) {
            Ok(_) => {
                let mut output_frame = Audio::empty();
                output_frame.set_rate(
                    reader
                        .target_sample_rate
                        .unwrap_or(reader.source_sample_rate),
                );
                output_frame.set_format(frame.format());
                output_frame.set_channel_layout(frame.channel_layout());
                output_frame.set_channels(frame.channels());

                let new_samples = (frame.samples() as u64
                    * reader
                        .target_sample_rate
                        .unwrap_or(reader.source_sample_rate) as u64
                    / frame.rate() as u64) as usize;
                output_frame.set_samples(new_samples);

                match resampler.run(&frame, &mut output_frame) {
                    Ok(_) => {
                        if output_frame.is_planar() {
                            let samples_per_channel = output_frame.samples();
                            for i in 0..samples_per_channel {
                                if reader.force_mono {
                                    let mut sample_sum = 0.0f32;
                                    for c in 0..output_frame.channels() as usize {
                                        sample_sum += output_frame.plane::<f32>(c)[i];
                                    }
                                    buffer.push(sample_sum / output_frame.channels() as f32);
                                } else {
                                    for c in 0..output_frame.channels() as usize {
                                        buffer.push(output_frame.plane::<f32>(c)[i]);
                                    }
                                }
                            }
                        } else {
                            if reader.force_mono {
                                let plane = output_frame.plane::<f32>(0);
                                for chunk in plane.chunks(output_frame.channels() as usize) {
                                    let avg = chunk.iter().sum::<f32>() / chunk.len() as f32;
                                    buffer.push(avg);
                                }
                            } else {
                                buffer.extend_from_slice(output_frame.plane::<f32>(0));
                            }
                        }
                        Ok(())
                    }
                    Err(_) => {
                        let new_resampler = Context::get(
                            decoder.format(),
                            decoder.channel_layout(),
                            reader.source_sample_rate,
                            decoder.format(),
                            decoder.channel_layout(),
                            reader
                                .target_sample_rate
                                .unwrap_or(reader.source_sample_rate),
                        )
                        .map_err(|e| AudioError::ResamplerCreation(e.to_string()))?;

                        *resampler = new_resampler;
                        Ok(())
                    }
                }
            }
            Err(ffmpeg_next::Error::Other { errno: _ }) => Ok(()),
            Err(e) => Err(AudioError::from(e)),
        }?;

        // Get next packet
        match input.packets().next() {
            Some((stream, packet)) => {
                if stream.index() != reader.stream_index {
                    Ok(())
                } else {
                    decoder.send_packet(&packet).map_err(AudioError::from)
                }
            }
            None => {
                let _ = decoder.send_eof();
                if buffer.is_empty() && unsafe { frame.is_empty() } {
                    break;
                }
                Ok(())
            }
        }?;
    }

    // Pre-allocate the exact size needed and drain buffer directly
    let samples_to_take = reader.chunk_size.min(buffer.len() / output_channels) * output_channels;
    if samples_to_take > 0 {
        let mut chunk = Vec::with_capacity(samples_to_take);
        chunk.extend(buffer.drain(..samples_to_take));
        Ok(chunk)
    } else {
        Ok(Vec::new())
    }
}

#[pyfunction]
#[pyo3(signature = (path, target_sample_rate=None, force_mono=false))]
fn load(
    path: String,
    target_sample_rate: Option<u32>,
    force_mono: bool,
    py: Python<'_>,
) -> PyResult<Py<PyArray2<f32>>> {
    let reader = AudioReader::new(path, target_sample_rate, 1024, force_mono)?;
    let total_samples = reader.__len__()?;

    // Decode all samples at once since we know the total length
    reader.decode_samples(total_samples, py)
}

#[pyfunction]
#[pyo3(signature = (path, data, sample_rate))]
fn write(path: String, data: &Bound<'_, PyArray2<f32>>, sample_rate: u32, py: Python<'_>) -> PyResult<()> {
    // Get array info before releasing GIL
    let array = data.readonly();
    let array = if !array.is_contiguous() {
        array.as_array().to_owned()
    } else {
        array.as_array().to_owned()
    };
    let num_samples = array.shape()[0];
    let num_channels = array.shape()[1];

    // Release GIL for the actual writing process
    py.allow_threads(move || {
        ffmpeg::init().map_err(|e| AudioError::FFmpeg(e.to_string()))?;

        let mut output = ffmpeg::format::output(&path).map_err(|e| AudioError::FFmpeg(e.to_string()))?;
        
        let codec_id = output.format().codec(&path, ffmpeg::media::Type::Audio);
        let global_header = output
            .format()
            .flags()
            .contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER);

        let codec = ffmpeg::encoder::find(codec_id)
            .ok_or_else(|| AudioError::FFmpeg("Encoder not found".to_string()))?
            .audio()
            .map_err(|e| AudioError::FFmpeg(e.to_string()))?;

        let channel_layout = ffmpeg::channel_layout::ChannelLayout::default(num_channels.try_into()?);
        
        let sample_format = codec
            .formats()
            .expect("unknown supported formats")
            .next()
            .unwrap();

        let mut stream = output
            .add_stream(codec)
            .map_err(|e| AudioError::FFmpeg(e.to_string()))?;

        stream.set_time_base((1, sample_rate.try_into()?));

        let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| AudioError::FFmpeg(e.to_string()))?;
        let mut encoder = context
            .encoder()
            .audio()
            .map_err(|e| AudioError::FFmpeg(e.to_string()))?;

        if global_header {
            encoder.set_flags(ffmpeg::codec::flag::Flags::GLOBAL_HEADER);
        }

        encoder.set_rate(sample_rate.try_into()?);
        encoder.set_channel_layout(channel_layout);
        encoder.set_channels(channel_layout.channels());
        encoder.set_format(sample_format);
        encoder.set_time_base((1, sample_rate.try_into()?));
        encoder.set_bit_rate(320_000);

        let mut encoder = encoder
            .open_as(codec)
            .map_err(|e| AudioError::FFmpeg(e.to_string()))?;

        stream.set_parameters(&encoder);

        let in_time_base = encoder.time_base();
        let out_time_base = stream.time_base();

        output
            .write_header()
            .map_err(|e| AudioError::FFmpeg(e.to_string()))?;

        let frame_size = encoder.frame_size() as usize;
        let mut frame = ffmpeg::frame::Audio::new(
            sample_format,
            frame_size,
            channel_layout,
        );
        frame.set_rate(sample_rate.try_into()?);
        frame.set_channels(num_channels.try_into()?);
        frame.set_format(sample_format);

        let mut encoded = ffmpeg::Packet::empty();
        let mut pts = 0i64;

        // Main encoding loop
        for chunk_start in (0..num_samples).step_by(frame_size) {
            let chunk_size = std::cmp::min(frame_size, num_samples - chunk_start);
            if chunk_size == 0 {
                break;
            }

            frame.set_pts(Some(pts));
            frame.set_samples(chunk_size);

            if sample_format.is_planar() {
                for channel in 0..num_channels {
                    let frame_data = frame.plane_mut::<f32>(channel);
                    frame_data[..chunk_size].copy_from_slice(
                        &array.slice(s![chunk_start..chunk_start + chunk_size, channel])
                            .to_owned()
                            .as_slice()
                            .unwrap()
                    );
                }
            } else {
                let frame_data = frame.plane_mut::<f32>(0);
                for i in 0..chunk_size {
                    for channel in 0..num_channels {
                        frame_data[i * num_channels + channel] = array[[chunk_start + i, channel]];
                    }
                }
            }

            encoder.send_frame(&frame).map_err(AudioError::from)?;
            pts += chunk_size as i64;

            receive_and_process_packets(&mut encoder, &mut encoded, &mut output, 0, in_time_base, out_time_base)?;
        }

        encoder.send_eof().map_err(AudioError::from)?;
        receive_and_process_packets(&mut encoder, &mut encoded, &mut output, 0, in_time_base, out_time_base)?;

        output.write_trailer().map_err(AudioError::from)?;
        Ok(())
    })
}

// Helper function to handle packet receiving and writing
fn receive_and_process_packets(
    encoder: &mut ffmpeg::encoder::Audio,
    packet: &mut ffmpeg::Packet,
    output: &mut ffmpeg::format::context::Output,
    stream_index: usize,
    in_time_base: ffmpeg::Rational,
    out_time_base: ffmpeg::Rational,
) -> Result<(), AudioError> {
    while encoder.receive_packet(packet).is_ok() {
        packet.set_stream(stream_index);
        packet.rescale_ts(in_time_base, out_time_base);
        packet.write_interleaved(output)
            .map_err(|e| AudioError::FFmpeg(e.to_string()))?;
    }
    Ok(())
}

#[pymodule]
fn chunkloader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(write, m)?)?;
    m.add_class::<AudioReader>()?;
    Ok(())
}
