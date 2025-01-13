use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use rubato::{Resampler, SincFixedIn, WindowFunction};
use rubato::SincInterpolationParameters;
use std::sync::Mutex;

#[pyclass]
struct AudioReader {
    file_path: String,
    target_sample_rate: u32,
    buffer: Mutex<Vec<f32>>,
    buffer_pos: Mutex<usize>,
    decoder: Mutex<Option<Box<dyn symphonia::core::codecs::Decoder>>>,
    format: Mutex<Option<Box<dyn symphonia::core::formats::FormatReader>>>,
    resampler: Mutex<Option<SincFixedIn<f32>>>,
    current_sample_rate: Mutex<u32>,
    channels: Mutex<usize>,
}

#[pymethods]
impl AudioReader {
    #[new]
    fn new(file_path: String, target_sample_rate: u32) -> PyResult<Self> {
        Ok(AudioReader {
            file_path,
            target_sample_rate,
            buffer: Mutex::new(Vec::new()),
            buffer_pos: Mutex::new(0),
            decoder: Mutex::new(None),
            format: Mutex::new(None),
            resampler: Mutex::new(None),
            current_sample_rate: Mutex::new(0),
            channels: Mutex::new(0),
        })
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    // Returns number of audio channels
    #[getter]
    fn get_channels(&self) -> usize {
        *self.channels.lock().unwrap()
    }

    // Returns the current sample rate
    #[getter]
    fn get_sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    fn __next__(slf: Bound<'_, Self>) -> PyResult<Option<(Vec<f32>, Vec<f32>)>> {
        let this = slf.borrow();
        if this.decoder.lock().unwrap().is_none() {
            this.initialize()?;
        }

        match this.decode_next_packet()? {
            Some(interleaved) => {
                let mut left = Vec::with_capacity(interleaved.len() / 2);
                let mut right = Vec::with_capacity(interleaved.len() / 2);
                
                for chunk in interleaved.chunks_exact(2) {
                    left.push(chunk[0]);
                    right.push(chunk[1]);
                }
                
                Ok(Some((left, right)))
            },
            None => Ok(None),
        }
    }

    // Get a single chunk of interleaved audio
    fn get_next_chunk(slf: Bound<'_, Self>) -> PyResult<Option<Vec<f32>>> {
        let this = slf.borrow();
        if this.decoder.lock().unwrap().is_none() {
            this.initialize()?;
        }

        this.decode_next_packet()
    }
}

impl AudioReader {
    fn initialize(&self) -> PyResult<()> {
        // Create media source
        let file = File::open(&self.file_path).map_err(|e| {
            PyValueError::new_err(format!("Failed to open file: {}", e))
        })?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create format reader
        let mut hint = Hint::new();
        if let Some(ext) = Path::new(&self.file_path).extension() {
            hint.with_extension(ext.to_str().unwrap_or(""));
        }

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
            .map_err(|e| PyValueError::new_err(format!("Failed to probe media format: {}", e)))?;

        let mut format = probed.format;
        let track = format.default_track().ok_or_else(|| {
            PyValueError::new_err("No default track found")
        })?;

        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| PyValueError::new_err(format!("Failed to create decoder: {}", e)))?;

        *self.current_sample_rate.lock().unwrap() = track.codec_params.sample_rate.unwrap_or(44100);
        *self.channels.lock().unwrap() = track.codec_params.channels.map(|c| c.count()).unwrap_or(2);

        // Verify stereo audio
        if *self.channels.lock().unwrap() != 2 {
            return Err(PyValueError::new_err(format!(
                "File must be stereo (has {} channels)", *self.channels.lock().unwrap()
            )));
        }

        // Initialize resampler if needed
        if *self.current_sample_rate.lock().unwrap() != self.target_sample_rate {
            let resampler = SincFixedIn::<f32>::new(
                self.target_sample_rate as f64 / *self.current_sample_rate.lock().unwrap() as f64,
                2.0,
                SincInterpolationParameters {
                    window: WindowFunction::BlackmanHarris2,
                    oversampling_factor: 256,
                    sinc_len: 64,
                    f_cutoff: 0.95,
                    interpolation: rubato::SincInterpolationType::Linear,
                },
                8192,
                *self.channels.lock().unwrap(),
            ).map_err(|e| PyValueError::new_err(format!("Failed to create resampler: {}", e)))?;

            let mut resampler_guard = self.resampler.lock().unwrap();
            *resampler_guard = Some(resampler);
        }

        let mut decoder_guard = self.decoder.lock().unwrap();
        let mut format_guard = self.format.lock().unwrap();

        *decoder_guard = Some(decoder);
        *format_guard = Some(format);

        Ok(())
    }

    fn decode_next_packet(&self) -> PyResult<Option<Vec<f32>>> {
        let mut format_guard = self.format.lock().unwrap();
        let mut decoder_guard = self.decoder.lock().unwrap();
        
        let format = format_guard.as_mut().unwrap();
        let decoder = decoder_guard.as_mut().unwrap();

        // Read next packet
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            Err(e) => return Err(PyValueError::new_err(format!("Error reading packet: {}", e))),
        };

        // Decode packet
        let decoded = decoder.decode(&packet).map_err(|e| {
            PyValueError::new_err(format!("Failed to decode packet: {}", e))
        })?;

        // Convert to f32 samples
        let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples().to_vec();

        // Get the number of channels once to avoid multiple locks
        let num_channels = *self.channels.lock().unwrap();

        // Resample if needed
        if let Some(resampler) = self.resampler.lock().unwrap().as_mut() {
            let chunks: Vec<Vec<f32>> = samples
                .chunks_exact(num_channels)
                .map(|c| c.to_vec())
                .collect();

            let resampled = resampler.process(&chunks.iter().map(|v| v.as_slice()).collect::<Vec<_>>(), None)
                .map_err(|e| PyValueError::new_err(format!("Failed to resample: {}", e)))?;

            // Flatten channels back to interleaved format
            let mut result = Vec::new();
            for i in 0..resampled[0].len() {
                for channel in 0..num_channels {
                    result.push(resampled[channel][i]);
                }
            }
            Ok(Some(result))
        } else {
            Ok(Some(samples))
        }
    }
}

#[pymodule]
fn audio_reader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AudioReader>()?;
    Ok(())
}