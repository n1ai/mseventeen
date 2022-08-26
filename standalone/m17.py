#!/usr/bin/env python3
################################################################################
# M17 Program: Implements the M17 Data Link Layer (and some of the PHY too)
################################################################################
# This work is licensed under the Creative Commons Attribution-ShareAlike
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# 
# This work is attributed to Rob Riggs, WX9O, Mobilinkd LLC.
# 
# The original sources of this work are:
# https://github.com/mobilinkd/m17-demodulator/blob/master/m17-modulator.ipynb
# https://github.com/mobilinkd/m17-demodulator/blob/master/m17-demodulator.ipynb
# 
# The source material has been adapted by David Cherkus, N1AI, starting on 1
# June 2022.  It has been modified from Juyper Notebook format to standalone
# Python format, with various adaptations as deemed desirable.
################################################################################

import numpy as np
from commpy.channelcoding import Trellis, conv_encode, viterbi_decode
import soundcard as sc
import pycodec2
import struct
import sys
import time
import binascii
import pprint
import io
import struct
import argparse
from argparse import RawTextHelpFormatter

# TODO: BERT mode, packet mode, symbol detection/generation, filtering, 4fsk
# TODO: implement input from microphone using soundcard.microphone class
# TODO: decide what should be per-frame vs per-class vs per-module variables
# TODO: split this file into python modules
# TODO: use threading so tx, rx, and audio output can be in separate threads


#### M17 Basic Utilities #######################################################
class M17Util(object):

    ## Basic Formatting Methods ################################################
    @staticmethod
    def int16_to_bytearray(f):
        return bytearray([(f>>8) & 0xFF, f & 0xFF])

    @staticmethod
    def bytes_to_bitarray(byte_array):
        return np.concatenate(\
            [[int((x & (1<<i)) != 0) \
              for i in range(7,-1,-1)] \
              for x in byte_array])

    @staticmethod
    def bits_to_bytes(bits):
        x = 0
        for i in bits:
            x <<= 1
            x |= i
        return x

    @staticmethod
    def bitarray_to_bytes(bits):
        return bytearray([M17Util.bits_to_bytes(x) \
                          for x in np.split(bits, len(bits)//8)])

    ## Debug Formatting Methods ################################################
    @staticmethod
    def bytes_to_hex(byte_array):
        return binascii.hexlify(byte_array)

    @staticmethod
    def bitarray_to_hex(bits):
        assert(len(bits)%8==0) # need bits to map to bytes
        b = M17Util.bitarray_to_bytes(bits)
        return binascii.hexlify(b)

    @staticmethod
    def show_bitarray(tag, bits):
        """an accurate hex representation of a bit array"""
        print("\n%s=%s %d" % (tag, M17Util.bitarray_to_hex(bits), len(bits)))

    @staticmethod
    def show_bytearray(tag, ba):
        """an accurate hex representation of a bytearray"""
        print("\n%s=%s %d" % (tag, M17Util.bytes_to_hex(ba), len(ba)*8))

    @staticmethod
    def show_estimates(tag, ea):
        """an INaccurate hex representation of an estimate array"""
        # it is inaccurate since depunctured bits are tri-valued.
        # this representation allows rough comparisons of estimates
        # in that they will be inaccurate in the same way
        ba = np.concatenate([[1] if e >= 0.0 else [0] for e in ea])
        M17Util.show_bitarray(tag,ba)

    @staticmethod
    def dump_audio(tag, d):
        """an accurate hex representation of audio samples"""
        assert(isinstance(d[0], np.int16))
        b = bytearray([])
        for x in d:
            b += M17Util.int16_to_bytearray(x)
        print("\n%s=%s" % (tag, M17Util.bytes_to_hex(b)))


    ## I/O Formatting Methods ##################################################
    @staticmethod
    def dibits_to_packed_symbols(dibits):
        """ packs four dibits into an unsigned 8-bit value in a bytearray """
        return bytearray([x[0]<<6|x[1]<<4|x[2]<<2|x[3]<<0
            for x in np.split(dibits, len(dibits)//4)])
    
    @staticmethod
    def dibits_to_unpacked_symbols(dibits):
        """ represents each dibit using its signed 8-bit value in bytearray """
        """ bytearray elements are unsigned so we use the signed bit pattern """
        symbol_map = [ 0x01, 0x03, 0xff, 0xfd ]
        return bytearray([symbol_map[x] for x in dibits])
    
    @staticmethod
    def bitarray_to_dibits(bits):
        # 1/0
        return np.array([(x[0]<<1|x[1]) for x in np.split(bits, len(bits)//2)])

    ## I/O Deformatting Methods ################################################
    @staticmethod
    def packed_symbols_to_dibits(pdibits):
        """ input is bytes with packed dibits """
        """ e.g. bytearray(b'\xb4') """
        """ output is dibits """
        l = []
        for b in pdibits:
            l.append((b&(3<<6))>>6)
            l.append((b&(3<<4))>>4)
            l.append((b&(3<<2))>>2)
            l.append((b&(3<<0))>>0)
        return l
    
    @staticmethod
    def unpacked_symbols_to_dibits(binsyms):
        """ input is bytes with symbols """
        """ e.g. bytearray(b'\x03\x01\xfd\xff') """
        """ output is dibits """
        # workaround for zeros at end of "m17-mod -s" output: ignore zeros
        symbol_dict = { 0x01:0, 0x03:1, 0xff:2, 0xfd:3, 0x00: -1}
        return [ symbol_dict[b] for b in binsyms if b != 0x00 ]
    
    @staticmethod
    def dibits_to_bitarray(dibits):
        return np.concatenate([[(d&0x2)>>1, d&0x1] for d in dibits])


## M17FileFormatter ###########################################################
class M17FileFormatter(object):
    """ changes to/from input and output formats based on cmd line args """
    def __init__(self, args):
        # similar logic is in both ReadAhead and M17FileFormatter
        if args.function == "loop":
            # when doing loopback everything stays internal so no formatting
            self.fmt = "none"
        elif args.bin:
            # if user has specified bin, it takes precedence
            self.fmt = "bin"
        else:
            # all other cases take default
            self.fmt = "sym"

    def format(self, bits):
        if self.fmt == "none":
            return bits
        dibits = M17Util.bitarray_to_dibits(bits)
        if self.fmt == "sym":
             return M17Util.dibits_to_unpacked_symbols(dibits)
        else:  # bin
             return M17Util.dibits_to_packed_symbols(dibits)

    def deformat(self, blob):
        if self.fmt == "none":
            return bits
        if self.fmt == "sym":
             dibits = M17Util.unpacked_symbols_to_dibits(blob)
        else:  # bin
             dibits = M17Util.packed_symbols_to_dibits(blob)
        return M17Util.dibits_to_bitarray(dibits)


#### Randomizing / Derandomizing ###############################################
class Randomize(object):
    def __init__(self):
        self.DC = [0xd6, 0xb5, 0xe2, 0x30, 0x82, 0xFF, 0x84, 0x62,
                   0xba, 0x4e, 0x96, 0x90, 0xd8, 0x98, 0xdd, 0x5d,
                   0x0c, 0xc8, 0x52, 0x43, 0x91, 0x1d, 0xf8, 0x6e,
                   0x68, 0x2F, 0x35, 0xda, 0x14, 0xea, 0xcd, 0x76,
                   0x19, 0x8d, 0xd5, 0x80, 0xd1, 0x33, 0x87, 0x13,
                   0x57, 0x18, 0x2d, 0x29, 0x78, 0xc3]
        self.dc_bits = np.concatenate([
             [int((x & (1<<i)) != 0) for i in range(7,-1,-1)] 
             for x in self.DC])

    def randomize(self, frame):
        result = np.bitwise_xor(frame, self.dc_bits)
        return result

    def derandomize(self, frame):
        result = np.bitwise_xor(frame, self.dc_bits)
        return result


#### M17 Randomizer / Derandomizer #############################################
class M17Randomizer(object):
    def __init__(self):
        self.randomizer = Randomize()

    def randomize(self, interleaved_bits):
        randomized_bits = self.randomizer.randomize(interleaved_bits)
        return(randomized_bits)

    def derandomize(self, channel_bits):
        derandomized_bits = self.randomizer.derandomize(channel_bits)
        return(derandomized_bits)


#### Interleaving / Deinterleaving #############################################
class PolynomialInterleaver(object):
    """Polynomial bit interleaver.  Default to 80.
    Valid LTE polynomials can be found here:
    https://github.com/supermihi/lpdec/blob/master/lpdec/codes/interleaver.py#L264
    """
    
    def __init__(self, f1 = 11, f2 = 20, k = 80):
        self.f1 = f1
        self.f2 = f2
        self.K = k
    
    def index(self, i):
        return ((self.f1 * i) + (self.f2 * i * i)) % self.K
    
    def interleave(self, data):
        result = np.zeros(self.K, dtype = data.dtype)
        for i in range(len(data)):
            result[self.index(i)] = data[i]
        return result
    
    def deinterleave(self, data):
        result = np.zeros(len(data), dtype = data.dtype)
        for i in range(len(data)):
            if i == self.K: break
            result[i] = data[self.index(i)]
        return result

    def interlv(self, data):
        return self.interleave(data)

    def deinterlv(self, data):
        return self.deinterleave(data)

#### M17 Interleaver / Deinterleaver ##########################################
class M17Interleaver(object):
    def __init__(self):
        self.interleaver = PolynomialInterleaver(45, 92, 368)

    def interleave(self, punctured_bits):
        interleaved_bits = self.interleaver.interlv(punctured_bits)
        return(interleaved_bits)

    def deinterleave(self, derandomized_bits):
        deinterleaved_bits = self.interleaver.deinterlv(derandomized_bits)
        return(deinterleaved_bits)

#### M17 Frame Combiner / Decombiner ###########################################
class M17FrameCombiner(object):
    def __init__(self):
        self.asrt = False
        pass

    def combine(self, lch_bits, str_bits):
        if self.asrt: assert(len(lch_bits) == 96)
        if self.asrt: assert(len(str_bits) == 272)
        cmb_bits = np.concatenate([lch_bits, str_bits])
        return(cmb_bits)

    def decombine(self, cmb_bits):
        if self.asrt: assert(len(cmb_bits) == 368)
        lch_bits = cmb_bits[0:96]
        str_bits = cmb_bits[96:]
        return(lch_bits, str_bits)


#### Puncturing / Depuncturing #################################################
class Puncturing(object):

    def puncturing(self, 
        message: np.ndarray, 
        punct_vec: np.ndarray) -> np.ndarray:
        """
        XXX: Applying of the punctured procedure.
        Parameters
        ----------
        message : 1D ndarray
            Input message {0,1} bit array.
        punct_vec : 1D ndarray
            Puncturing vector {0,1} bit array.
        Returns
        -------
        punctured : 1D ndarray
            Output punctured vector {0,1} bit array.
        """
        shift = 0
        N = len(punct_vec)
        punctured = []
        for idx, item in enumerate(message):
            if punct_vec[idx % N] == 1:
                punctured.append(item)
        return np.array(punctured)

    def depuncturing(self, 
        punctured: np.ndarray, 
        punct_vec: np.ndarray, 
        shouldbe: int) -> np.ndarray:
        """
        Applying of the inserting zeros procedure.
        Parameters
        ----------
        punctured : 1D ndarray
            Input punctured message {0,1} bit array.
        punct_vec : 1D ndarray
            Puncturing vector {0,1} bit array.
        shouldbe : int 
            Length of the initial message (before puncturing).
        Returns
        -------
        depunctured : 1D ndarray
            Output vector {0,1} bit array.
        """
        shift2 = 0
        N = len(punct_vec)
        depunctured = np.zeros((shouldbe,))
        for idx, item in enumerate(depunctured):
            if punct_vec[idx % N] == 1:
                depunctured[idx] = float(punctured[idx - shift2])
            else:
                shift2 = shift2 + 1
        return depunctured
    

#### M17 Stream Puncturer / Depuncturer ########################################
class M17StrPuncturer(object):
    def __init__(self):
        self.puncturer = Puncturing()
        self.P2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

    def puncture(self, encoded_bits):
        punctured_bits = self.puncturer.puncturing(encoded_bits, self.P2)
        return(punctured_bits)

    def depuncture(self, punctured_bits):
        # Shift/scale the bits from [1,0] to [-1,+1]
        punctured_bits = np.array(punctured_bits) * 2 - 1
        depunctured_bits = self.puncturer.depuncturing(punctured_bits,
            self.P2, 296)
        return(depunctured_bits)


#### M17 Encoder / Decoder ####################################################
class M17Encoder(object):
    def __init__(self):
        self.memory = np.array([4])
        self.trellis = Trellis(self.memory, np.array([[0o31,0o27]]))

    def encode(self, subframe_bits):
        # The M17 spec says we need to add four bits to the payload to flush
        # the encoder, but when we use 'term' the encoder does this for us
        encoded_bits = conv_encode(subframe_bits, self.trellis, 'term')
        # The encoder produces the eight bits that represent the encoded tail 
        # bits and the decoder needs them to decode so we send those along
        return(encoded_bits)

    def decode(self, encoded_bits, depunctured=True):
        # The input to the soft decoder are LLR values (log likelihood ratios).
        # +1.0 means likely to be binary 1, -1.0 means likely to be binary 0,
        # 0.0 means equally likely to be binary 0 or 1.  The data we get from
        # the depuncturer uses 0.0 to indicate where the data was punctured
        # which helps the decoder when it tries to recover the original data.
        # The data we get when depunctured is False needs to be mapped here
        # using +1.0 for binary 1, -1.0 for binary 0,
        if not depunctured:
            encoded_bits = np.concatenate(
                [[1.0] if b == 1 else [-1.0] for b in encoded_bits])
        # 1: use ints rather than floats for soft decode
        # input_bits = np.zeros(len(encoded_bits), dtype=np.int)
        # input_bits = np.concatenate(
        #     [[500] if b == 1 else [-500] for b in encoded_bits])
        # decoded_bits = viterbi_decode(input_bits, self.trellis, 
        #     decoding_type='soft', tb_depth=20)
        # 2: try hard decode
        # input_bits = np.zeros(len(encoded_bits), dtype=np.int)
        # input_bits = np.concatenate(
        #     [[1] if b == 1 else [0] for b in encoded_bits])
        # decoded_bits = viterbi_decode(input_bits, self.trellis, 
        #     decoding_type='hard', tb_depth=20)
        # 3: back to original
        decoded_bits = viterbi_decode(encoded_bits, self.trellis, 
            decoding_type='soft', tb_depth=20)
        # The tail/flush bits generated during modulation can now be dropped
        decoded_bits = decoded_bits[0:len(decoded_bits)-4]
        return(decoded_bits)


#### M17 Stream SubFramer ######################################################
class M17StrSubFramer(object):
    def __init__(self):
        self.FN = 0

    def enframe(self, frame, last_frame):
        # convert current frame number to byte array
        if last_frame:
            fn = M17Util.int16_to_bytearray(self.FN|0x8000)
        else:
            fn = M17Util.int16_to_bytearray(self.FN)
        # update the frame number
        self.FN += 1
        if self.FN == 0x8000:
            self.FN = 0 
        # create new frame to be FEC-encoded
        subframe_bytes = fn + frame
        subframe_bits = M17Util.bytes_to_bitarray(subframe_bytes)
        # return bits to be encoded
        return(subframe_bits)

    def deframe(self, subframe_bits):
        # convert subframe bits to byes
        data = M17Util.bitarray_to_bytes(subframe_bits)
        fn_demod = (data[0] << 8) | data[1]
        # print("\nFN = %d" % fn_demod)
        # recover frame
        frame = data[2:]
        return(frame)


#### M17 Stream Modulator ######################################################
class M17StrModulator(object):
    def __init__(self, lchmod, show=False, asrt=False):
        self.lchmod = lchmod  # lchmod object already has lsf bits
        self.show = show      # should we show debug stuff?
        self.asrt = asrt      # should we make assertions?
        self.subframer = M17StrSubFramer()
        self.encoder = M17Encoder()
        self.puncturer = M17StrPuncturer()
        self.interleaver = M17Interleaver()
        self.combiner = M17FrameCombiner()
        self.randomizer = M17Randomizer()

    def modulate(self, data, last_frame):
        if self.asrt: assert(len(data) == 128//8)
        if self.show: M17Util.show_bytearray("Dm",data)

        # Create stream subframe with FN and data
        subframe_bits = self.subframer.enframe(data, last_frame)
        if self.asrt: assert(len(subframe_bits) == 144)
        if self.show: M17Util.show_bitarray("Fm",subframe_bits)

        # Encode the subframe using our FEC encoder
        encoded_bits = self.encoder.encode(subframe_bits)
        if self.asrt: assert(len(encoded_bits) == 296)
        if self.show: M17Util.show_bitarray("Em",encoded_bits)

        # Puncture the data
        punctured_bits = self.puncturer.puncture(encoded_bits)
        if self.asrt: assert(len(punctured_bits) == 272)
        if self.show: M17Util.show_bitarray("Pm",punctured_bits)

        # Form the LICH bits for the LCH subchannel
        lch_bits = self.lchmod.get_next_lch()

        # Combine LCH and STR to form a frame
        if self.asrt: assert(np.count_nonzero(lch_bits)==0)
        combined_bits = self.combiner.combine(lch_bits,punctured_bits)
        if self.asrt: assert(len(combined_bits) == 368)

        # Interleave the data
        interleaved_bits = self.interleaver.interleave(combined_bits)
        if self.asrt: assert(len(interleaved_bits) == 368)
        if self.show: M17Util.show_bitarray("Im",interleaved_bits)

        # Randomize the data
        randomized_bits = self.randomizer.randomize(interleaved_bits)
        if self.asrt: assert(len(randomized_bits) == 368)
        if self.show: M17Util.show_bitarray("Rm",randomized_bits)

        # Randomization is final stage of modulation so return the result
        return(randomized_bits)


#### M17 Stream Demodulator ####################################################
class M17StrDemodulator(object):
    def __init__(self, lchdem, lsfdem, show=False, asrt=True):
        self.lchdem = lchdem  # link information (sub-)channel demodulator
        self.lsfdem = lsfdem  # lsf demodulator that decodes lch data
        self.show = show      # should we show debug stuff?
        self.asrt = asrt      # should we make assertions?
        self.subframer = M17StrSubFramer()
        self.encoder = M17Encoder()
        self.puncturer = M17StrPuncturer()
        self.interleaver = M17Interleaver()
        self.combiner = M17FrameCombiner()
        self.randomizer = M17Randomizer()

    def demodulate(self, randomized_bits):
        # Randomization is the final stage of modulation 
        # Therefore demodulation starts with derandomization

        # De-randomize data to recover interleaved bits
        if self.asrt: assert(len(randomized_bits) == 368)
        if self.show: M17Util.show_bitarray("Rd",randomized_bits)
        interleaved_bits = self.randomizer.derandomize(randomized_bits)

        # De-interleave data to recover combined bits
        if self.asrt: assert(len(interleaved_bits) == 368)
        if self.show: M17Util.show_bitarray("Id",interleaved_bits)
        combined_bits = self.interleaver.deinterleave(interleaved_bits)
        if self.asrt: assert(len(combined_bits) == 368)

        # De-combine data to recover LCH and punctured stream bits
        lch_bits, punctured_bits = self.combiner.decombine(combined_bits)
        if self.asrt: assert(len(lch_bits) == 96)
        if self.asrt: assert(len(punctured_bits) == 272)
        if self.show: M17Util.show_bitarray("Pd",punctured_bits)

        # Pass LCH sub-channel data to the LCH demodulator
        # It will return a full LSF when it has reassembled one
        lsf_bits = self.lchdem.decode_lch(lch_bits)
        if lsf_bits is not None:
            # Decode the LSF made from the LCH sub-channel
            lsf = M17Util.bitarray_to_bytes(lsf_bits)
            ok = self.lsfdem.decode_lsf("lch", lsf)

        # De-puncture data to recover encoded bits
        # Note that de-punctured data contains estimated bits
        encoded_bits = self.puncturer.depuncture(punctured_bits)
        if self.asrt: assert(len(encoded_bits) == 296)
        if self.show: M17Util.show_estimates("Ed",encoded_bits)

        # De-encode data to recover stream subframe
        # When testing w/o depuncturing, set depunctured flag to false...
        # decoded_bits = self.encoder.decode(encoded_bits, depunctured=False)
        decoded_bits = self.encoder.decode(encoded_bits)
        if self.asrt: assert(len(decoded_bits) == 144)
        if self.show: M17Util.show_bitarray("Fd",decoded_bits)

        # Recover payload from stream subframe
        data = self.subframer.deframe(decoded_bits)
        if self.asrt: assert(len(data) == 128//8)
        if self.show: M17Util.show_bytearray("Dd",data)
        # DC: HACK: show demod output only 
        # M17Util.show_bytearray("Dd",data)

        # Return demodulated stream data
        return(data)

#### CRC16 calculator (only used by LSF/LCH, not STR) ##########################
class CRC16(object):
    def __init__(self, poly, init):
        self.poly = poly
        self.init = init
        self.mask = 0xFFFF
        self.msb = 0x8000
        self.reset()
    
    def reset(self):
        self.reg = self.init
        for i in range(16):
            bit = self.reg & 0x01
            if bit:
                self.reg ^= self.poly
            self.reg >>= 1
            if bit:
                self.reg |= self.msb
        self.reg &= self.mask

    def crc(self, data):
        for byte in data:
            for i in range(8):
                msb = self.reg & self.msb
                self.reg = ((self.reg << 1) & self.mask) | ((byte >> (7 - i)) & 0x01)
                if msb:
                    self.reg ^= self.poly
         
    def get(self):
        reg = self.reg
        for i in range(16):
            msb = reg & self.msb
            reg = ((reg << 1) & self.mask)
            if msb:
                reg ^= self.poly
        return reg & self.mask
    
    def get_bytes(self):
        crc = self.get()
        return bytearray([(crc>>8) & 0xFF, crc & 0xFF])


#### M17 LSF Puncturer / Depuncturer ##########################################
class M17LsfPuncturer(object):
    def __init__(self):
        self.puncturer = Puncturing()
        self.P1 = [1] + [1, 0, 1, 1] * 15

    def puncture(self, encoded_bits):
        punctured_bits = self.puncturer.puncturing(encoded_bits, self.P1)
        return(punctured_bits)

    def depuncture(self, punctured_bits):
        # Shift/scale the bits from [1,0] to [-1,+1]
        punctured_bits = np.array(punctured_bits) * 2 - 1
        depunctured_bits = self.puncturer.depuncturing(punctured_bits,
            self.P1, 488)
        return(depunctured_bits)


#### Golay24 ##################################################################
class Golay24(object):
    
    POLY = 0xC75
    POLY_a = np.array([1,1,0,0,0,1,1,1,0,1,0,1], dtype=np.int)
    
    @staticmethod
    def syndrome(codeword):
        for i in range(12):
            if codeword & 1:
                codeword ^= Golay24.POLY
            codeword >>= 1
        return codeword

    @staticmethod
    def popcount(data):
        count = 0
        for i in range(24):
            count += (data & 1)
            data >>= 1
        return count
    
    @staticmethod
    def parity(data):
        return Golay24.popcount(data) & 1
    
    def __init__(self):
        # Construct the syndrome-keyed correction lookup table.
        self.LUT = {}
        for error in self._make_3bit_errors(23):
            syn = self.syndrome(error)
            self.LUT[syn] = error

    def encode23(self, bits):
        codeword = bits;
        for i in range(12):
            if codeword & 1:
                codeword ^= Golay24.POLY
            codeword >>= 1
        return codeword | (bits << 11)

    def encode(self, bits):
        codeword = self.encode23(bits)
        return (codeword << 1) | self.parity(codeword)

    def encode_array(self, bits):
        data = 0
        for bit in bits:
            data = (data << 1) | bit
        codeword = self.encode23(data)
        encoded = (codeword << 1) | self.parity(codeword)
        result = np.zeros(24, dtype=int)
        for i in range(24):
            result[23 - i] = encoded & 1
            encoded >>= 1
        
        return result
    
    def decode(self, bits):
        syndrm = self.syndrome(bits >> 1);
        try:
            correction = self.LUT[syndrm]
            errors = self.popcount(correction)
            corrected = bits ^ (correction << 1)
            if (errors < 3) or not self.parity(corrected):
                return corrected, errors
            else:
                return None, 4
        except KeyError:
            return None, 4

    def decode_array(self, bits):
        data = 0
        for bit in bits:
            data = (data << 1) | bit
        decoded, errors = self.decode(data)
        if decoded is None:
            return decoded, errors
        result = np.zeros(24, dtype=int)
        for i in range(24):
            result[23 - i] = decoded & 1
            decoded >>= 1
        
        return result, errors

    @staticmethod
    def _make_3bit_errors(veclen=24):
        """Return a list of all bitvectors with <= 3 bits as 1's.
        This returns list of lists, each 24 bits long by default.
        """
        errorvecs = []
        # all zeros
        errorvecs.append(0)
        # one 1
        for i in range(veclen):
            errorvecs.append(1 << i)
        # two 1s
        for i in range(veclen - 1):
            for j in range(i + 1, veclen):
                errorvecs.append((1 << i) | (1 << j))
        # three 1s
        for i in range(veclen  - 2):
            for j in range(i + 1, veclen - 1):
                for k in range(j + 1, veclen):
                    errorvecs.append((1 << i) | (1 << j) | (1 << k))
        return errorvecs


#### M17 LCH Modulator #########################################################
class M17LchModulator(object):
    def __init__(self, lsf, show=False, asrt=False):
        self.show = show
        self.asrt = asrt
        self.golay = Golay24()
        self.lch_list = self.make_lch(lsf)
        self.lch_cnt = 0

    def mod_enc_lsf_cnk(self, lsf_cnk_bits):
        """ modulator-side code to encode one lsf chunk """
        if self.asrt: assert(len(lsf_cnk_bits==48))
        # cut lsf chunk into four 12-bit lch cut blocks
        lch_cut_list = np.split(lsf_cnk_bits, 4)
        # encode each lch_cut
        lch_enc_bits = []
        for i, lch_cut in enumerate(lch_cut_list):
            lch_enc_bits.append(self.golay.encode_array(lch_cut))
            if self.asrt: assert(len(lch_enc_bits[i]==24))
        # join the cuts into one np array and return them
        lch_enc_bits = np.concatenate(lch_enc_bits)
        if self.asrt: assert(len(lch_enc_bits==96))
        return(lch_enc_bits)

    def make_lch (self, lsf_bytes):
        if self.show: M17Util.show_bytearray("lsf", lsf_bytes)
        lsf_all_bits = M17Util.bytes_to_bitarray(lsf_bytes)
        # split lsf up into six cuts, each of five bytes or 40 bits
        lsf_cut_list = np.split(lsf_all_bits, 6)
        lsf_enc_list = []
        # loop through the lsf chunks in the list
        for i, lsf_cut_bits in enumerate(lsf_cut_list):
            if self.asrt: assert(len(lsf_cut_bits) == 40)
            # add CNT/RSVD byte to each of the six lsf chunks
            lsf_cnk_bits = np.concatenate([lsf_cut_bits, 
                M17Util.bytes_to_bitarray([i << 5])])
            if self.asrt: assert(len(lsf_cnk_bits) == 48)
            # encode the lsf chunk
            lch_enc_bits = self.mod_enc_lsf_cnk(lsf_cnk_bits)
            if self.asrt: assert(len(lch_enc_bits) == 96)
            if self.show: M17Util.show_bitarray("mod_enc_lch_%d" % i, 
                lch_enc_bits)
            # save encoded bits to the list
            lsf_enc_list.append(lch_enc_bits)
        # return list of golay-encoded chunks
        return(lsf_enc_list)

    def get_next_lch(self):
        # each STR frame needs a LCH 
        next_lch = self.lch_list[self.lch_cnt]
        self.lch_cnt += 1
        if self.lch_cnt > 5:
            self.lch_cnt = 0
        return(next_lch)

 
#### M17 LCH Demodulator #######################################################
class M17LchDemodulator(object):
    def __init__(self, show=False, asrt=False):
        self.show = show  # should we show debug stuff?
        self.asrt = asrt  # should we make assertions?
        self.golay = Golay24() 
        self.saved_lch_bits = None
        self.lch_cnt = 0  # debug only, not operational

    def dem_dec_lch_enc(self, lch_enc_bits):
        """ demodulator-side code to decode one lch encoded block """
        # cut lsf enc block into four 24-bit lch enc cuts
        lch_cut_list = np.split(lch_enc_bits, 4)
        # decode each lch_cut
        lch_dec_bits = []
        for i, lch_cut in enumerate(lch_cut_list):
            # CRC protects the LCH so error count is not checked here
            # Might want to log the case where err != 0 some day
            dec, err = self.golay.decode_array(lch_cut)
            if self.asrt: assert(dec is not None and len(dec==24))
            lch_dec_bits.append(dec[:12])
        # join the cuts into one np array and return them
        lsf_dec_bits = np.concatenate(lch_dec_bits)
        return(lsf_dec_bits)

    def decode_lch(self, lch_enc_bits):
        """ receiver is presenting lch demod with 96 lch bits """
        if self.asrt: assert(len(lch_enc_bits)==96)
        if self.show: M17Util.show_bitarray(("dem_enc_lch_%d" % self.lch_cnt),
            lch_enc_bits)
        lsf_dec_bits = self.dem_dec_lch_enc(lch_enc_bits)
        if self.asrt: assert(len(lsf_dec_bits)==48)
        if self.show: M17Util.show_bitarray("dem_dec_lch_%d"%self.lch_cnt,
            lsf_dec_bits)
        self.lch_cnt += 1
        # put the 40 lsf bits (without the cnt/rsvd byte) into lch decoder
        # returns the lch bits when 240 have been gathered
        # TODO: use the cnt byte to know when we've gathered a full LCH
        return self.put_next_lch(lsf_dec_bits[:40])

    def put_next_lch(self, lch_bits):
        if self.asrt: assert(len(lch_bits) == 40)
        if self.saved_lch_bits is None:
            self.saved_lch_bits = lch_bits
            return None # not long enough to decode as a LSF
        elif len(self.saved_lch_bits) < 200: 
            self.saved_lch_bits = np.concatenate([self.saved_lch_bits, 
              lch_bits])
            return None # not long enough to decode as a LSF
        elif len(self.saved_lch_bits) == 200: 
            self.saved_lch_bits = np.concatenate([self.saved_lch_bits, 
              lch_bits])
            return self.saved_lch_bits
        elif len(self.saved_lch_bits) == 240: 
            self.saved_lch_bits = np.concatenate([self.saved_lch_bits[40:], 
              lch_bits])
            return self.saved_lch_bits
        else:
            raise ValueError("bug in lch reassembly!")
        # exit here if you want to debug just the lch reassembly
        # sys.exit(0)


#### M17 LSF Modulator #########################################################
class M17LsfModulator(object):
    def __init__(self, callsign, show=False, asrt=False):
        self.show = show  # should we show debug stuff?
        self.asrt = asrt  # should we make assertions?
        # lsf doesn't change so make it once from constructor
        self.lsf = self.make_lsf(callsign)
        # lsf doesn't change so modulate it once from constructor
        self.lsf_mod = self.modulate_lsf(self.lsf)

    def get_lsf(self):
        """ allow the lch modulator to get the lsf so it can make lch """
        return(self.lsf)

    def encode_callsign_base40(self, callsign):
        # Encode the characters to base-40 digits.
        encoded = 0;
        for c in callsign[::-1]:
            encoded *= 40;
            if c >= 'A' and c <= 'Z':
                encoded += ord(c) - ord('A') + 1
            elif c >= '0' and c <= '9':
                encoded += ord(c) - ord('0') + 27
            elif c == '-':
                encoded += 37
            elif c == '/':
                encoded += 38
            elif c == '.':
                encoded += 39
            else:
                pass # invalid
        # Convert the integer value to a byte array.
        result = bytearray()
        for i in range(6):
            result.append(encoded & 0xFF)
            encoded >>= 8
        # Reverse the byte order for big-endian transmit
        result.reverse()
        return result

    def make_lsf (self, callsign):
        # encode the given callsign
        encoded_call = self.encode_callsign_base40(callsign)
        # create broadcast address
        broadcast = bytearray([0xff] * 6)
        # create frame type:
        #   bit 0:      0=pkt, 1=str
        #   bits 2-3:   00=rsvd, 01=data, 10=voice, 11=voice+data
        #   bits 3-4:   00=no encryption, 01=aes, 10=scramble, 11=rsvd
        #   bits 5-6:   encryption subtype, based on bits 3-4
        #   bits 7-10:  channel access number (CAN)
        #   bits 11-15: rsvd
        frame_type = bytearray([0x00,0x05])
        # create meta: it's all zeros since we are not doing encryption
        meta = bytearray([0x00]*14)
        # calculate the LSF's CRC
        crc = CRC16(0x5935, 0xFFFF)
        crc.reset()
        crc.crc(encoded_call)
        crc.crc(broadcast)
        crc.crc(frame_type)
        crc.crc(meta)
        cval = crc.get_bytes()
        # create the LSF by joining all its parts: DST SRC TYPE META CRC
        lsf = np.concatenate([encoded_call, broadcast, frame_type, meta, cval])
        return(lsf)

    def modulate_lsf(self, data):
        self.encoder = M17Encoder()
        self.puncturer = M17LsfPuncturer()
        self.interleaver = M17Interleaver()  # common...
        self.randomizer = M17Randomizer()       # common...

        # Check LSF frame size and contents
        if self.asrt: assert(len(data) == 240//8)
        if self.show: M17Util.show_bytearray("Dm",data)

        # Encode the frame using our FEC encoder
        frame_bits = M17Util.bytes_to_bitarray(data)
        encoded_bits = self.encoder.encode(frame_bits)
        if self.asrt: assert(len(encoded_bits) == 488)
        if self.show: M17Util.show_bitarray("Em",encoded_bits)

        # Puncture the data
        punctured_bits = self.puncturer.puncture(encoded_bits)
        if self.asrt: assert(len(punctured_bits) == 368)
        if self.show: M17Util.show_bitarray("Pm",punctured_bits)

        # Interleave the data
        interleaved_bits = self.interleaver.interleave(punctured_bits)
        if self.asrt: assert(len(interleaved_bits) == 368)
        if self.show: M17Util.show_bitarray("Im",interleaved_bits)

        # Randomize the data
        randomized_bits = self.randomizer.randomize(interleaved_bits)
        if self.asrt: assert(len(randomized_bits) == 368)
        if self.show: M17Util.show_bitarray("Rm",randomized_bits)

        # Randomization is final stage of modulation so return the result
        return(randomized_bits)

    def modulate(self):
        # return modulated bits to be sent
        return(self.lsf_mod)
 

#### M17 LSF Demodulator #######################################################
class M17LsfDemodulator(object):
    def __init__(self, show=False, asrt=False):
        # TODO: should we instantiate a LSF decoder for each rx session?
        #   - we won't know the src/dst till a full LSF is decoded
        #   - we should keep passing in LSFs till we get a decode
        self.show = show  # should we show debug stuff?
        self.asrt = asrt  # should we make assertions?
        self.encoder = M17Encoder()
        self.puncturer = M17LsfPuncturer()      # NOT common
        self.interleaver = M17Interleaver()
        self.randomizer = M17Randomizer()

    def demodulate(self, randomized_bits):
        # Randomization is the final stage of modulation 
        # Therefore demodulation starts with derandomization

        # De-randomize data to recover interleaved bits
        if self.asrt: assert(len(randomized_bits) == 368)
        if self.show: M17Util.show_bitarray("Rd",randomized_bits)
        interleaved_bits = self.randomizer.derandomize(randomized_bits)

        # De-interleave data to recover combined bits
        if self.asrt: assert(len(interleaved_bits) == 368)
        if self.show: M17Util.show_bitarray("Id",interleaved_bits)
        punctured_bits = self.interleaver.deinterleave(interleaved_bits)
        if self.asrt: assert(len(punctured_bits) == 368)

        # De-puncture data to recover encoded bits
        # Note that de-punctured data contains estimated bits
        encoded_bits = self.puncturer.depuncture(punctured_bits)
        if self.asrt: assert(len(encoded_bits) == 488)
        if self.show: M17Util.show_estimates("Ed",encoded_bits)

        # De-encode data to recover stream subframe
        # When testing w/o depuncturing, set depunctured flag to false...
        # decoded_bits = self.encoder.decode(encoded_bits, depunctured=False)
        decoded_bits = self.encoder.decode(encoded_bits)
        if self.asrt: assert(len(decoded_bits) == 240)
        if self.show: M17Util.show_bitarray("Fd",decoded_bits)

        # Convert frame bits to bytes
        data = M17Util.bitarray_to_bytes(decoded_bits)
        if self.asrt: assert(len(data) == 240//8)
        if self.show: M17Util.show_bytearray("Dd",data)

        # Return demodulated lsf data
        return(data)

    def decode_callsign_base40(self, encoded_bytes):
        # check for special case of broadcast
        if encoded_bytes == bytearray([0xff] * 6):
            return "broadcast"
        # Convert byte array to integer value.
        encoded_bytes.reverse()
        i,h = struct.unpack("IH", encoded_bytes)
        encoded = (h << 32) | i
        # print('{:#012x}'.format(encoded))
        # Unpack each base-40 digit and map them to the appriate character.
        result = io.StringIO()
        while encoded:
            result.write(
              "xABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/."[encoded % 40])
            encoded //= 40;
        return result.getvalue();

    def decode_lsf(self, tag, lsf):
        if self.asrt: assert(len(lsf) == 240//8)
        self.crc = CRC16(0x5935, 0xFFFF)
        self.crc.crc(lsf)
        self.checksum = self.crc.get()
        self.crc_ok = self.checksum == 0
        #if self.show or self.crc_ok:   #  print if show is set or crc is OK
        if False: # NOTE: bypass printing for now
            dst = self.decode_callsign_base40(lsf[:6])
            src = self.decode_callsign_base40(lsf[6:12])
            print('%s: dst: %s src: %s crc: %s' % 
                 (tag, src, dst, "ok" if self.crc_ok else "BAD"), 
                 file = sys.stderr)
        return(self.crc_ok)


#### M17 Transmitter ###########################################################
class M17Transmitter(object):
    # TODO: transmit syncwords
    def __init__(self, callsign):
        # instantiate lsfmod first since it builds the lsf needed for lch
        self.lsfmod = M17LsfModulator(callsign)
        # instantiate lchmod next since lsf is subchannel of str
        # provide lchmod with the lsf formed by lsfmod
        lchmod = M17LchModulator(self.lsfmod.get_lsf())
        # provide lchmod to strmod since lch is a subchannel of str 
        self.strmod = M17StrModulator(lchmod)
        # preamble in bit form 
        self.preamble = M17Util.bytes_to_bitarray([0x77]*48)
        # eot in bit form 
        self.eot = M17Util.bytes_to_bitarray([0x55, 0x5D]*24)
        # lsf syncword in bit form
        self.lsf_sw = M17Util.bytes_to_bitarray([0x55, 0xF7])
        # str syncword in bit form
        self.str_sw = M17Util.bytes_to_bitarray([0xFF, 0x5D])

    def send_preamble(self):
        # note: tx level data is bits not bytes.
        return(self.preamble)

    def start(self):
        # create the LSF frame
        mframe = self.lsfmod.modulate()
        # send the LSF sync word and the LSF frame
        return(np.concatenate([self.lsf_sw, mframe]))

    def transmit(self, data, last_frame):
        # create the STR frame (streaming data)
        mframe = self.strmod.modulate(data, last_frame)
        # send the STR sync word and the STR frame
        mframe = np.concatenate([self.str_sw, mframe])
        return(mframe)

    def stop(self):
        # send EOT frame
        return(self.eot)


#### M17 Receiver ##############################################################
class M17Receiver(object):
    # TODO: receive syncwords
    def __init__(self):
        # lsfdem is used at this level and in strdem to decode lsfs
        self.lsfdem = M17LsfDemodulator()
        # lchdem is used by strdem to decode lch info
        # could choose to instantiate it in strdem instead
        # for now, keep object instantiation here just like tx
        lchdem = M17LchDemodulator()
        self.strdem = M17StrDemodulator(lchdem, self.lsfdem)
        # preamble in bit form 
        self.preamble = M17Util.bytes_to_bitarray([0x77]*48)
        # eot in bit form 
        self.eot = M17Util.bytes_to_bitarray([0x55, 0x5D]*24)
        # lsf syncword in bit form
        self.lsf_sw = M17Util.bytes_to_bitarray([0x55, 0xF7])
        # str syncword in bit form
        self.str_sw = M17Util.bytes_to_bitarray([0xFF, 0x5D])

    # TODO: Need a wimpy rx loop that looks at first few bytes of
    # incoming frame to determine if this is sync, lsf, str, pkt, brt
    # or eot and act accordingly.  Also need to return an indication
    # of eot back to high level

    def recv_preamble(self,mframe):
        # Check that the preamble bits are the ones we expect
        np.testing.assert_array_equal(mframe, self.preamble, "Preamble mismatch")
        return(mframe)

    def start(self, mframe):
        # check that the LSF sw is as expected
        np.testing.assert_array_equal(mframe[:16], self.lsf_sw, "LSF mismatch")
        # remove LSF sync word
        mframe = mframe[16:]
        # recv LSF frame
        dframe = self.lsfdem.demodulate(mframe)
        # decode LSF frame
        self.lsfdem.decode_lsf("lsf", dframe)
        return(dframe)

    def receive(self, mframe):
        # check that the STR sw is as expected
        np.testing.assert_array_equal(mframe[:16], self.str_sw, "STR mismatch")
        # remove STR sync word
        mframe = mframe[16:]
        # recv STR frame(s)
        dframe = self.strdem.demodulate(mframe)
        return(dframe)

    def stop(self, mframe):
        # Check that the eot bits are the ones we expect.
        # Some versions of m17-mod in symbol mode mess up the EOT.
        # It puts out a syncword's worth of the correct bytes
        # [ 03 03 03 03 03 03 FD 03 ] followed by 40 zeros so 48 bytes.  
        # Fix is at https://github.com/n1ai/m17-cxx-demod/pull/1
        # For now, just issue a warning instead of raising an error.
        if not np.array_equal(mframe, self.eot):
            print("Warning: EOT is not in expected form", file = sys.stderr)
        # recv EOT frame
        return(mframe)

#### ReadAhead ##############################################################
class ReadAhead(object):
    """ read ahead by one block so we can set last frame flag """
    """ eventually, we should do a queue with high water mark """
    """ so we can read from a live microphone instead of a file """
    def __init__(self, args, C2C):
        # similar 'args' logic is in both ReadAhead and M17FileFormatter...
        # m17 block sizes:
        # - m17 has 384 bits per block or 192 symbols (di-bits) per block 
        # - for sym format, 192 syms * 1 byte / 1 sym = 192 bytes per block
        # - for bin format, 192 syms * 1 byte / 4 sym =  48 bytes per block
        if args.function == "loop" or args.function == "tx":
            # loop and transmit functions read audio samples 
            self.blocksize = C2C.C2_IBPR
        elif args.encoding == "m17":
            if args.bin:
                # m17 receiver reading packed dibits
                self.blocksize = 48
            else:
                # m17 receiver reading symbols
                self.blocksize = 192
        else:   # non-m17 receivers
            # read enough samples to produce two audio frames
            self.blocksize = C2C.C2_OBPR
        self.infile = args.infile
        self.curr = self.__readblock()
        if len(self.curr) == 0:  
            raise ValueError("input file has no data")
        self.prev = self.curr 

    def __readblock(self):
        ipacket = self.infile.read(self.blocksize)
        if len(ipacket) > 0 and len(ipacket) < self.blocksize:
            # if there some data but less than a block, zero fill to full block
            ipacket += bytearray([0]*(self.blocksize-len(ipacket)))
        assert(len(ipacket)==0 or len(ipacket)==self.blocksize)
        return(ipacket)

    def get_next_block(self):
        self.curr = self.__readblock()
        # four possibilities:
        #   prev=empty  curr=empty  => can't happen if flag is honored
        #   prev=empty  curr=full   => can't happen since we read in order
        #   prev=full   curr=empty  => last_frame is True
        #   prev=full   curr=full   => last_frame is False
        if len(self.prev) == 0 and len(self.curr) == 0:
            raise ValueError("readahead buffers are empty")
        elif len(self.prev) == 0 and len(self.curr) != 0:
            raise ValueError("readahead out of order")
        elif len(self.prev) != 0 and len(self.curr) == 0:
            # current read returns zero bytes => done
            data = self.prev       # use data from last time
            self.prev = self.curr  # save new data for next time
            return data, True
        elif len(self.prev) != 0 and len(self.curr) != 0:
            # current read returns full block => !done
            data = self.prev       # use data from last time
            self.prev = self.curr  # save new data for next time
            return data, False
        else:
            # should not happen...
            raise ValueError("programming error")

#### Command Line Argument Parsing #############################################
M17desc=r"""
By default, reads audio from the given file, encodes/decodes it using M17's
encoding/decoding scheme, and plays the audio to the default speaker.  Can also
create a M17 stream using "tx" function or play one using "rx" function.  These
streams can be in either sym (default) or bin format.  Encoding can be M17
(default), Codec2 (to demonstrate Codec2 encoding without M17 encoding)  or
None (to demonstrate audio playback without Codec2 or M17 encoding)"""

M17expl=r"""
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

$ m17 --infile audio.sym --function rx 

"""

def get_command_line_args(): 
    parser = argparse.ArgumentParser(description=M17desc, epilog=M17expl,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("-i", "--infile", default=sys.stdin,
                        type=argparse.FileType('rb'),
                        help="input file (default is '-' for STDIN).")
    parser.add_argument("-o", "--outfile", default=sys.stdout,
                        type=argparse.FileType('wb'),
                        help="output file (default is '-' for STDOUT).")
    parser.add_argument("-f", "--function", default="loop",
                        choices=["loop", "tx", "rx"],
                        help="function to perform (default is loop).")
    # format options (bin vs sym) are mutually exclusive.  bin may be specified
    # and if it is the parser returns true for both but the code priortizes bin
    # over sym since specification suggests it should take precedence
    fgroup = parser.add_mutually_exclusive_group()
    fgroup.add_argument("-b", "--bin", action="store_true", default=False,
                        help="use packed dibit format (default is sym).")
    fgroup.add_argument("-s", "--sym",  action="store_true", default=True,
                        help="use symbol format (default).")
    # end format group
    parser.add_argument("-e", "--encoding", default="m17",
                        choices=["m17", "codec2", "none"],
                        help="encoding to use (default is m17).")
    parser.add_argument("-c", "--callsign", default="N0CAL",
                        help="callsign to send (default is N0CAL).")
    parser.add_argument("-S", "--speaker", action="store_true",
                        help="use speaker for output (default is False).")
    # "mode" is not implemented yet
    # parser.add_argument("-m", "--mode", default="str",
    #                     choices=["str", "pkt", "brt"],
    #                     help="mode to operate in (default is str).")

    # create the argument parser
    args = parser.parse_args()

    if hasattr(args.infile, 'buffer'):
        #   If the argument was '-', FileType('rb') ignores the 'b' when
        #   wrapping stdin. Fix that by grabbing the underlying binary reader.
        args.infile = args.infile.buffer

    if hasattr(args.outfile, 'buffer'):
        #   If the argument was '-', FileType('wb') ignores the 'b' when
        #   wrapping stdout. Fix that by grabbing the underlying binary writer.
        args.outfile = args.outfile.buffer

    return args

## Codec2 Configuration ########################################################
class Codec2Configuration(object):
    """ Codec2 related configuration stuff, for easy sharing """
    def __init__(self):
        self.C2_ISR  = 8000                              # input sample rate
        self.C2_OBR  = 3200                              # output bit rate 
        self.C2 = pycodec2.Codec2(self.C2_OBR)           # c2 object itself
        self.C2_encode = self.C2.encode                  # convenience
        self.C2_decode = self.C2.decode                  # convenience
        self.C2_IBPS = 2                                 # in bytes per sample
        self.C2_ISPF = int(self.C2.samples_per_frame())  # in samples per frame
        self.C2_IFMT = '{}h'.format(2*self.C2_ISPF)      # in halfwords per read
        self.C2_IBPF = int(self.C2_ISPF*self.C2_IBPS)    # in bytes per frame
        self.C2_IBPR = int(2*self.C2_IBPF)               # in bytes per read 
        self.C2_OBPF = int(self.C2.bits_per_frame()/8)   # out bytes per frame
        self.C2_OBPR = int(2*self.C2_OBPF)               # out bytes per read 
    def __repr__(self):                                  # print all the stuff
        return str(self.__dict__)

## Loop Function ###############################################################
def LoopFunction(args, C2C):
    # create a file formatter
    fmt = M17FileFormatter(args)

    # create a reader too 
    reader = ReadAhead(args, C2C)

    # If using m17, set up stream modulator and demodulator objects
    if args.encoding == "m17":
        # Set up M17 transmitter and receive objects
        m17tx = M17Transmitter(args.callsign)
        m17rx = M17Receiver()
        #
        # NOTE: The following is based on perfect ordering of M17 frames.
        # It should be enhanced to be state-driven instead...
        #
        # Send M17 preamble
        ptx = m17tx.send_preamble()                # generate the preamble
        # Receive M17 preamble
        prx = m17rx.recv_preamble(ptx)             # rx the preamble
        # For now, use start() to send the LSF
        mframe = m17tx.start()                     # generate the lsf
        dframe = m17rx.start(mframe)                  # rx the lsf
    
    # When using the speaker, we batch to amortize the cost of playing sound.
    # We play roughly one second of sound per batch.  A better approch would
    # be coprocessing or multiprocessing but I want to keep this program simple.
    batch = 0
    if args.speaker:
        default_speaker = sc.default_speaker()
        BATCH_SIZE = int(C2C.C2_ISR/(C2C.C2_IBPF))
    else: 
        BATCH_SIZE = 1  # write audio as soon as it is generated

    # Common loop to process the data in the input file, be it aud, bin or sym
    audio = np.array([], dtype=np.int16)
    while True:
        # handle m17 last frame as a special case
        m17_last_frame = False
        # read a block of data (blocksize depends on input encoding)
        ipacket, last_frame = reader.get_next_block()
        # for the loop function, the input is sixteen bit audio samples
        npacket = np.array(struct.unpack(C2C.C2_IFMT,ipacket), dtype=np.int16)
        # test the encoding to be used
        if args.encoding == "none":
            audio = np.concatenate([audio, npacket])
        elif args.encoding == "codec2":
            # input is sixteen bit audio samples
            encode0 = C2C.C2.encode(npacket[0:C2C.C2_ISPF])
            encode1 = C2C.C2.encode(npacket[C2C.C2_ISPF:])
            # output is audio from decoded codec2 samples 
            decode0 = C2C.C2.decode(encode0)
            decode1 = C2C.C2.decode(encode1)
            audio = np.concatenate([audio, decode0, decode1])
        elif args.encoding == "m17":
            # c2 at rate 3200 takes in 160 two-byte samples
            # c2 at rate 3200 produces eight bytes (64 bits)
            # m17 stream mode takes 128 bits as payload per packet
            # thus we pack two c2 outputs into one m17 packet
            encode0 = C2C.C2.encode(npacket[0:C2C.C2_ISPF])
            encode1 = C2C.C2.encode(npacket[C2C.C2_ISPF:])
            mframe = m17tx.transmit(encode0 + encode1, last_frame)
            # m17 last_frame is eot, don't handle it here
            if last_frame:
                m17_last_frame = True
            else:
                # demodulated frame contains two codec2 frames
                dframe = m17rx.receive(mframe)
                assert(len(dframe)==C2C.C2_OBPR)
                encode0 = dframe[0:C2C.C2_OBPF]
                encode1 = dframe[C2C.C2_OBPF:]
                decode0 = C2C.C2.decode(bytes(encode0))
                decode1 = C2C.C2.decode(bytes(encode1))
                audio = np.concatenate([audio, decode0, decode1])
        else:
            raise ValueError("unknown encoder type")

        # common output code for the most common output case, audio samples,
        # add this packet to the batch
        if not m17_last_frame:
            batch += 1
            if not args.speaker:  
                # TODO: some sort of agc / normalization?
                args.outfile.write(struct.pack(C2C.C2_IFMT, *audio))  # OUCH!
                batch = 0
                audio = np.array([], dtype=np.int16)
            elif batch == BATCH_SIZE: # time to play the audio?
                audio = audio / np.max(audio)  # as per soundcard webpage
                default_speaker.play(audio, samplerate=C2C.C2_ISR)
                batch = 0
                audio = np.array([], dtype=np.int16)

        # last frame just processed?
        if last_frame:
            # check for batched audio not yet played
            if args.speaker and len(audio) > 0:
                audio = audio / np.max(audio)  # as per soundcard webpage
                default_speaker.play(audio, samplerate=C2C.C2_ISR)
                batch = 0
                audio = np.array([], dtype=np.int16)
            break
    
    # If using m17, handle EOT frame 
    # NOTE: This assumes perfect ordering of M17 frames.
    # It should be enhanced to be state-driven instead...
    if args.encoding == "m17":
        # For now, use stop() to send and receive the EOT
        # We do this so our data is compatible with other programs
        mframe = m17tx.stop()                      # generate the eot
        dframe = m17rx.stop(mframe)                # rx the eot from stop()


## TX Function ###############################################################
def TransmitFunction(args, C2C):
    # create a file formatter
    fmt = M17FileFormatter(args)

    # create a reader too 
    reader = ReadAhead(args, C2C)

    # If using m17, set up stream modulator and demodulator objects
    if args.encoding == "m17":
        # Set up M17 transmitter and receive objects
        m17tx = M17Transmitter(args.callsign)
        #
        # NOTE: The following is based on perfect ordering of M17 frames.
        # It should be enhanced to be state-driven instead...
        #
        # Send M17 preamble
        ptx = m17tx.send_preamble()                # generate the preamble
        pwr = fmt.format(ptx)                      # format it for tx
        args.outfile.write(pwr)                    # write it for tx
        # For now, use start() to send the LSF
        mframe = m17tx.start()                     # generate the lsf
        wframe = fmt.format(mframe)                # format it for tx
        args.outfile.write(wframe)                 # write it for tx
    
    # When using the speaker, we batch to amortize the cost of playing sound.
    # We play roughly one second of sound per batch.  A better approch would
    # be coprocessing or multiprocessing but I want to keep this program simple.
    batch = 0
    if args.speaker:
        default_speaker = sc.default_speaker()
        BATCH_SIZE = int(C2C.C2_ISR/(C2C.C2_IBPF))
    else: 
        BATCH_SIZE = 1  # write audio as soon as it is generated

    # Common loop to process the data in the input file, be it aud, bin or sym
    audio = np.array([], dtype=np.int16)
    while True:
        # read a block of data (blocksize depends on input encoding)
        ipacket, last_frame = reader.get_next_block()
        # for the tx function, the input is sixteen bit audio samples
        npacket = np.array(struct.unpack(C2C.C2_IFMT,ipacket), dtype=np.int16)
        # test the encoding to be used
        if args.encoding == "none":
            raise ValueError('"none" encoding must use the loop function')
        elif args.encoding == "codec2":
            # input is sixteen bit audio samples
            encode0 = C2C.C2.encode(npacket[0:C2C.C2_ISPF])
            encode1 = C2C.C2.encode(npacket[C2C.C2_ISPF:])
            # output is encoded codec2 samples stored in bytes
            args.outfile.write(encode0)
            args.outfile.write(encode1)
        elif args.encoding == "m17":
            # c2 at rate 3200 takes in 160 two-byte samples
            # c2 at rate 3200 produces eight bytes (64 bits)
            # m17 stream mode takes 128 bits as payload per packet
            # thus we pack two c2 outputs into one m17 packet
            encode0 = C2C.C2.encode(npacket[0:C2C.C2_ISPF])
            encode1 = C2C.C2.encode(npacket[C2C.C2_ISPF:])
            mframe = m17tx.transmit(encode0 + encode1, last_frame)
            # TODO: rationalize tx path writing right from main loop
            wframe = fmt.format(mframe) 
            args.outfile.write(wframe)
        else:
            raise ValueError("unknown encoder type")

        # last frame just processed?
        if last_frame:
            # check for batched audio not yet played
            if args.speaker and len(audio) > 0:
                audio = audio / np.max(audio)  # as per soundcard webpage
                default_speaker.play(audio, samplerate=C2C.C2_ISR)
                batch = 0
                audio = np.array([], dtype=np.int16)
            break
    
    # If using m17, handle EOT frame 
    # NOTE: This assumes perfect ordering of M17 frames.
    # It should be enhanced to be state-driven instead...
    if args.encoding == "m17":
        # For now, use stop() to send and receive the EOT
        # We do this so our data is compatible with other programs
            mframe = m17tx.stop()                      # generate the eot
            wframe = fmt.format(mframe)                # format it for tx
            args.outfile.write(wframe)                 # write it for tx


## RX Function ################################################################
def ReceiveFunction(args, C2C):
    # create a file formatter
    fmt = M17FileFormatter(args)

    # create a reader too 
    reader = ReadAhead(args, C2C)

    # If using m17, set up stream modulator and demodulator objects
    if args.encoding == "m17":
        if args.function == "loop" or args.function == "rx":
            m17rx = M17Receiver()
        #
        # NOTE: The following is based on perfect ordering of M17 frames.
        # It should be enhanced to be state-driven instead...
        #
        prd, last_frame = reader.get_next_block()  # read the preamble
        ptx = fmt.deformat(prd)                    # reformat for rx
        prx = m17rx.recv_preamble(ptx)             # rx the preamble
        rframe, last_frame = reader.get_next_block()  # read the lsf
        mframe = fmt.deformat(rframe)                 # reformat for rx
        dframe = m17rx.start(mframe)                  # rx the lsf
    
    # When using the speaker, we batch to amortize the cost of playing sound.
    # We play roughly one second of sound per batch.  A better approch would
    # be coprocessing or multiprocessing but I want to keep this program simple.
    batch = 0
    if args.speaker:
        default_speaker = sc.default_speaker()
        BATCH_SIZE = int(C2C.C2_ISR/(C2C.C2_IBPF))
    else: 
        BATCH_SIZE = 1  # write audio as soon as it is generated

    # Common loop to process the data in the input file, be it aud, bin or sym
    audio = np.array([], dtype=np.int16)
    while True:
        # handle m17 last frame as a special case
        m17_last_frame = False
        # read a block of data (blocksize depends on input encoding)
        ipacket, last_frame = reader.get_next_block()
        # test the encoding to be used
        if args.encoding == "none":
            # support the loop function for testing the common audio paths
            raise ValueError('"none" encoding must use the loop function')
        elif args.encoding == "codec2":
            # input is encoded codec2 samples stored in bytes
            encode0 = ipacket[0:C2C.C2_OBPF]
            encode1 = ipacket[C2C.C2_OBPF:]
            # output is audio from decoded codec2 samples 
            decode0 = C2C.C2.decode(encode0)
            decode1 = C2C.C2.decode(encode1)
            audio = np.concatenate([audio, decode0, decode1])
        elif args.encoding == "m17":
            mframe = fmt.deformat(ipacket)
            # m17 last_frame is eot, don't handle it here
            if last_frame:
                m17_last_frame = True
            else:
                dframe = m17rx.receive(mframe)
                # demodulated frame contains two codec2 frames
                assert(len(dframe)==C2C.C2_OBPR)
                encode0 = dframe[0:C2C.C2_OBPF]
                encode1 = dframe[C2C.C2_OBPF:]
                decode0 = C2C.C2.decode(bytes(encode0))
                decode1 = C2C.C2.decode(bytes(encode1))
                audio = np.concatenate([audio, decode0, decode1])
        else:
            raise ValueError("unknown encoder type")

        # m17 last_frame is eot, don't handle it here
        if not m17_last_frame: 
            # add this packet to the batch
            batch += 1
            if not args.speaker:  
                # TODO: some sort of agc / normalization?
                args.outfile.write(struct.pack(C2C.C2_IFMT, *audio))
                batch = 0
                audio = np.array([], dtype=np.int16)
            elif batch == BATCH_SIZE: # time to play the audio?
                audio = audio / np.max(audio)  # as per soundcard webpage
                default_speaker.play(audio, samplerate=C2C.C2_ISR)
                batch = 0
                audio = np.array([], dtype=np.int16)

        # last frame just processed?
        if last_frame:
            # check for batched audio not yet played
            if args.speaker and len(audio) > 0:
                audio = audio / np.max(audio)  # as per soundcard webpage
                default_speaker.play(audio, samplerate=C2C.C2_ISR)
                batch = 0
                audio = np.array([], dtype=np.int16)
            break
    
    # If using m17, handle EOT frame here.
    # NOTE: This assumes perfect ordering of M17 frames.
    # It should be enhanced to be state-driven instead...
    if args.encoding == "m17":
        # For now, use stop() to send and receive the EOT
        # We do this so our data is compatible with other programs
        # last frame is read in the loop above, make sure it is still around
        assert(last_frame)
        assert(mframe is not None)
        dframe = m17rx.stop(mframe)


################################################################################
#### Main Program ##############################################################
################################################################################

if __name__ == "__main__":
    # get command line args, defaults are: 
    #     args.infile:'-', 
    #     args.function:'loop', 
    #     args.bin:False, 
    #     args.sym:True, 
    #     args.encoding:'m17'
    args = get_command_line_args()

    # get codec2 configuration
    C2C = Codec2Configuration()

    # perform chosen function
    if args.function == "loop":
        LoopFunction(args, C2C)
    elif args.function == "tx":
        TransmitFunction(args, C2C)
    elif args.function == "rx":
        ReceiveFunction(args, C2C)

