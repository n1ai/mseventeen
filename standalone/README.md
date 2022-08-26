# M17 Python Standalone 

A Python implementation of the streaming mode of the 
[M17 Data Link Layer](https://spec.m17project.org/part-1/data-link-layer).

## Description

This folder/directory contains a standalone program that implements the
M17 data link layer in Python.

It is heavily based on the 
[M17 Modulator Notebook](https://github.com/mobilinkd/m17-demodulator/blob/master/m17-modulator.ipynb)
and the 
[M17 Demodulator Notebook](https://github.com/mobilinkd/m17-demodulator/blob/master/m17-demodulator.ipynb)
created by Rob Riggs, WX9O, Mobilinkd LLC.
 
This work reorganizes that code, makes it executable as a single program,
adds its own command line parser and help text, adds debugging messages
and assertions that can be enabled, and fixes a few small issues in that
code that are mostly due to that code being written to an older version
of the M17 specification.

The resulting single program provides transmit, receive and loopback modes.

It supports m17 stream encoding, codec2-only encoding, or no encoding at 
all.  The extra modes are useful for comparing the various levels of encoding
when studying voice quality or measuring performance.

This code was originally developed with the goal of it becoming the
basis for a GNU Radio Python block, but my initial results indicate
that while transmit performance was good, receive performance was not.
In particular, the overhead of the commpy viterbi decoder was not good
enough for real-time decoding of a single M17 stream on a fast Intel
CPU. This led me to decide to publish this work as it currently is and
to focus my future efforts on exploring C++ implementations.  I may or
may not revisit this code, but for now, I'm releasing it as-is in case
it is useful to others.

This code is tested against the M17 modulator/demodulator code in the 
[n1ai m17-cxx-demod repo](https://github.com/n1ai/m17-cxx-demod) which 
is a fork of the 
[mobilinkd m17-cxx-demod repo](https://github.com/mobilinkd/m17-cxx-demod).
The fork has cherry picked code from pending pull requests, in particular
the code from 
[Pull Request 21](https://github.com/mobilinkd/m17-cxx-demod/pull/21)
by [robojay](https://github.com/robojay)
that provides support for the bin, sym and rrc file formats that is needed
for the code in this repo to work as intended.

## Getting Started

### Dependencies

This code was developed on the Linux operating system using standard
Python3 and related development tools.  The
[radioconda](https://wiki.gnuradio.org/index.php/CondaInstall#Installation_using_radioconda) 
environment was used during development of this work because it provides
so many useful software tools for Python radio work, but any up to date 
Linux environment should suffice.

I know of no reason why this code would not work on other OSes with 
modern Python3 environments, but so far only Linux has been tested.

The following python packages will need to be installed, typically 
on Linux or Conda by using the `pip3` command:

* [commpy](https://pypi.org/project/scikit-commpy)
* [soundcard](https://pypi.org/project/SoundCard)
* [pycodec2](https://pypi.org/project/pycodec2)

These packages may have their own dependencies, e.g. pycodec2 
needs to have the Linux codec2 package installed for it to work.

### Installing

* Download the contents of this folder
* Optionally, use 'make test' to run the test script, which depends on 
  having the code from https://github.com/n1ai/m17-cxx-demod built and
  installed
* Optionally, use 'sudo make install' to install m17.py to /usr/local/bin

### Executing the program

The `m17test.sh` script shows several different examples of 
how to use this program.  That script uses files from the 
[n1ai m17-meta repo](https://github.com/n1ai/m17-meta)
that provide input audio, as well as m17 in `bin` and
`sym` format that can be processed by the `m17.py` program.

Also, there is a `--help` command line option that provides
detailed information on how to use the program:

```
$ ./m17.py --help
usage: m17.py [-h] [-i INFILE] [-o OUTFILE] [-f {loop,tx,rx}] [-b | -s]
              [-e {m17,codec2,none}] [-c CALLSIGN] [-S]

By default, reads audio from the given file, encodes/decodes it using M17's
encoding/decoding scheme, and plays the audio to the default speaker.  Can also
create a M17 stream using "tx" function or play one using "rx" function.  These
streams can be in either sym (default) or bin format.  Encoding can be M17
(default), Codec2 (to demonstrate Codec2 encoding without M17 encoding)  or
None (to demonstrate audio playback without Codec2 or M17 encoding)

optional arguments:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        input file (default is '-' for STDIN).
  -o OUTFILE, --outfile OUTFILE
                        output file (default is '-' for STDOUT).
  -f {loop,tx,rx}, --function {loop,tx,rx}
                        function to perform (default is loop).
  -b, --bin             use packed dibit format (default is sym).
  -s, --sym             use symbol format (default).
  -e {m17,codec2,none}, --encoding {m17,codec2,none}
                        encoding to use (default is m17).
  -c CALLSIGN, --callsign CALLSIGN
                        callsign to send (default is N0CAL).
  -S, --speaker         use speaker for output (default is False).

typical usage:

1) Read audio in signed sixteen bit little endian integer single channel 8000
   hertz format, encode using codec2 and M17, decode using M17 and code2, play 
   to default speaker.

$ m17.py --infile audio.aud 

2) Same as (1), with seperate transmitter and receiver for better performance.

$ m17.py --infile audio.aud --function tx | m17.py --function rx

3) Read audio in, encode using codec2 and M17, save output in M17 symbol format.

$ m17.py --infile audio.aud --function tx > audio.sym

4) Read audio in M17 symbol format, play to default speaker

$ m17.py --infile audio.sym --function rx 

```

## Possible Future Directions

Some ideas:
* Implement the parts of a full M17 stack that are not implemented here:
  BERT mode, packet mode, symbol detection/generation, filtering, 4fsk,
  "radio-like" state machines and queuing, etc
* Implement input from microphone using `soundcard.microphone` class
* Move away from the single file approach used during development to 
  Python modules
* Use Python threading so transmit, recive, and audio can be in separate 
  threads
* Find or make a faster implementation of a viterbi decoder

## Help

Reach out to `n1ai` on the 
[M17 Matrix or Discord channels](https://m17project.org/#contact).

## Authors

[David Cherkus, N1AI](https://github.com/n1ai)

## Version History

* 0.1
    * Initial Release

## License

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].
To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## Acknowledgments/Attributions

The main sources of this work are attributed to Rob Riggs, WX9O, Mobilinkd LLC.

These source are:
* [M17 Modulator Notebook](https://github.com/mobilinkd/m17-demodulator/blob/master/m17-modulator.ipynb)
* [M17 Demodulator Notebook](https://github.com/mobilinkd/m17-demodulator/blob/master/m17-demodulator.ipynb)
 
These sources have been adapted by David Cherkus, N1AI, starting on 1
June 2022.  They have been modified from Jupyter Notebook format to standalone
Python format, with various adaptations made as deemed desirable.

