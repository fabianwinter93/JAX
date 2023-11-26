from ...code.ChannelMixers.FastFeedForwardNetwork import FastFeedForwardNetwork, tests as tests_fffn
from ...code.TimeMixers.GateLoop import GateLoop, tests as tests_GL
from ...code.TimeMixers.LinearRecurrentUnit import LinearRecurrentUnit, tests as tests_LRU 
from ...code.TimeMixers.QLSTM import QLSTM, tests as tests_QLSTM
from ...code.TimeMixers.SuperHighPerformanceExtractor import SuperHighPerformanceExtractor, tests as tests_SHE


tests = [tests_fffn, tests_GL, tests_LRU, tests_QLSTM, tests_SHE]

layers = [FastFeedForwardNetwork.FastFeedForwardNetwork, 
          GateLoop.GateLoop, 
          LinearRecurrentUnit.LinearRecurrentUnit, 
          QLSTM.QLSTM, 
          SuperHighPerformanceExtractor.SuperHighPerformanceExtractor]

timemixers = [GateLoop.GateLoop, LinearRecurrentUnit.LinearRecurrentUnit, QLSTM.QLSTM, SuperHighPerformanceExtractor.SuperHighPerformanceExtractor]
channelmixers = [FastFeedForwardNetwork.FastFeedForwardNetwork]

