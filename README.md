Allows privatization and task-parallelization of decoding audio files across multiple threads. Loads audio files into numpy arrays of a desired size while allowing free threading, then yields chunks back to the GIL via the iterator protocol.

Probably terrible for most purposes. Still, good for deep learning. Which is what I do. :thumbsup:
