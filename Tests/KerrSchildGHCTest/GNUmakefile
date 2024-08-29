# -*- Mode: Makefile -*-

### This makefile produces an executable for each name in the `ebase'
###  variable using the libraries named in the `LibNames' variable.

# Included makefiles need an absolute path to the Chombo installation
# CHOMBO_HOME := Please set the CHOMBO_HOME locally(e.g. export CHOMBO_HOME=... in bash)

GHC_SOURCE = $(shell pwd)/../../Source

ebase := KerrSchildGHCTest

LibNames := BoxTools

src_dirs := $(GHC_SOURCE)/utils $(GHC_SOURCE)/GHC $(GHC_SOURCE)/BoxUtils $(GHC_SOURCE)/simd

include $(CHOMBO_HOME)/mk/Make.test
