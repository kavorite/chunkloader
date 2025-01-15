Allows privatization and task-parallelization of decoding audio files across multiple threads. Loads audio files into numpy arrays of a desired size while allowing free threading, then yields chunks back to the GIL via the iterator protocol.

Probably terrible for most purposes. Still, good for deep learning. Which is what I do. :thumbsup:

Once all the kinks have been worked out of free-threaded Python and it's been widely adopted and stabilized, I'll probably just use librosa, because FFmpeg is usually a little overkill. Libsndfile is not 'thread-safe,' but if you keep processing on each SNDFILE handle local to its own thread, you shouldn't need to deal with contention, similar to the way this module works.