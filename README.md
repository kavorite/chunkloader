Privatizes the work of parallelizing audio files across multiple threads. Loads audio files into numpy ararys of a desired size while allowing free threading, then yield back to the GIL via the iterator protocol.

Probably terrible for most purposes. Still, good for deep learning. Which is what I do. :thumbsup:
