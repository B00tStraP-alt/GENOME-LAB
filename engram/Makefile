# ==================================================================================================
# ENGRAM -- build.
#
#   make              Linux build of every test
#   make test         ... and run them
#   make win          Windows PE32+ build (mingw-w64, fully static: one EXE, no DLLs)
#   make wine-test    run the Windows build under Wine
#   make asan         AddressSanitizer + UndefinedBehaviorSanitizer build, and run it
#   make gate         THE PHASE GATE: all three, plus the self-check that the gate itself can fail
#   make clean
#
# Every variant builds with -Werror. A warning is a defect that has not been read yet.
#
# -ffp-contract=off IS NOT OPTIONAL. With it on, a compiler targeting an FMA-capable CPU may fuse
# a*b+c into one rounding instead of two, so the same source computes different bits on different
# machines -- and ENGRAM's projection and initialisation are recomputed rather than stored, so a store
# written on one machine would be read through different geometry on another (rule R5).
# ==================================================================================================

SRCDIR   := src
TESTDIR  := test
BUILD    := build

LIB_SRC  := $(wildcard $(SRCDIR)/*.c)
LIB_HDR  := $(wildcard $(SRCDIR)/*.h) $(wildcard $(TESTDIR)/*.h)
TESTS    := test_core test_text test_enc test_chunk test_align test_store

WARN := -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wcast-qual \
        -Wstrict-prototypes -Wmissing-prototypes -Wformat=2 -Wundef -Wvla -Wdouble-promotion \
        -Wwrite-strings -Wpointer-arith -Wnull-dereference -Wswitch-enum -Werror

BASE := -std=c99 -ffp-contract=off -fno-common $(WARN)

# ---- Linux ----------------------------------------------------------------------------------------
L_CC      := gcc
L_CFLAGS  := $(BASE) -O2 -g -pthread
L_LDFLAGS := -pthread -lm
L_OBJ     := $(patsubst $(SRCDIR)/%.c,$(BUILD)/linux/obj/%.o,$(LIB_SRC))
L_BIN     := $(addprefix $(BUILD)/linux/,$(TESTS))

# ---- Windows --------------------------------------------------------------------------------------
# __USE_MINGW_ANSI_STDIO: C99-conformant printf (%zu, %llu) instead of the legacy CRT's.
# -static: one self-contained EXE -- no libgcc, no libwinpthread beside it (commandment: one EXE).
W_CC      := x86_64-w64-mingw32-gcc
W_CFLAGS  := $(BASE) -O2 -D__USE_MINGW_ANSI_STDIO=1
W_LDFLAGS := -static -lbcrypt
W_OBJ     := $(patsubst $(SRCDIR)/%.c,$(BUILD)/win/obj/%.o,$(LIB_SRC))
W_BIN     := $(addsuffix .exe,$(addprefix $(BUILD)/win/,$(TESTS)))

# ---- Sanitizers -----------------------------------------------------------------------------------
A_CC      := gcc
A_CFLAGS  := $(BASE) -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
             -fno-sanitize-recover=all -pthread
A_LDFLAGS := -fsanitize=address,undefined -pthread -lm
A_OBJ     := $(patsubst $(SRCDIR)/%.c,$(BUILD)/asan/obj/%.o,$(LIB_SRC))
A_BIN     := $(addprefix $(BUILD)/asan/,$(TESTS))

# ---- ThreadSanitizer (cannot share a build with ASan) ---------------------------------------------
T_CC      := gcc
T_CFLAGS  := $(BASE) -O1 -g -fno-omit-frame-pointer -fsanitize=thread -pthread
T_LDFLAGS := -fsanitize=thread -pthread -lm
T_OBJ     := $(patsubst $(SRCDIR)/%.c,$(BUILD)/tsan/obj/%.o,$(LIB_SRC))
T_BIN     := $(addprefix $(BUILD)/tsan/,$(TESTS))

WINEPREFIX ?= $(abspath $(BUILD))/wineprefix
WINE_ENV   := WINEPREFIX=$(WINEPREFIX) WINEDEBUG=-all WINEDLLOVERRIDES="mscoree,mshtml="

.PHONY: all test win wine-test asan tsan plant gate clean
.SECONDARY:

all: $(L_BIN)

# ---- object rules ---------------------------------------------------------------------------------
$(BUILD)/linux/obj/%.o: $(SRCDIR)/%.c $(LIB_HDR) | $(BUILD)/linux/obj
	$(L_CC) $(L_CFLAGS) -c $< -o $@

$(BUILD)/win/obj/%.o: $(SRCDIR)/%.c $(LIB_HDR) | $(BUILD)/win/obj
	$(W_CC) $(W_CFLAGS) -c $< -o $@

$(BUILD)/asan/obj/%.o: $(SRCDIR)/%.c $(LIB_HDR) | $(BUILD)/asan/obj
	$(A_CC) $(A_CFLAGS) -c $< -o $@

$(BUILD)/tsan/obj/%.o: $(SRCDIR)/%.c $(LIB_HDR) | $(BUILD)/tsan/obj
	$(T_CC) $(T_CFLAGS) -c $< -o $@

# ---- test binaries --------------------------------------------------------------------------------
$(BUILD)/linux/%: $(TESTDIR)/%.c $(L_OBJ) $(LIB_HDR)
	$(L_CC) $(L_CFLAGS) -o $@ $< $(L_OBJ) $(L_LDFLAGS)

$(BUILD)/win/%.exe: $(TESTDIR)/%.c $(W_OBJ) $(LIB_HDR)
	$(W_CC) $(W_CFLAGS) -o $@ $< $(W_OBJ) $(W_LDFLAGS)

$(BUILD)/asan/%: $(TESTDIR)/%.c $(A_OBJ) $(LIB_HDR)
	$(A_CC) $(A_CFLAGS) -o $@ $< $(A_OBJ) $(A_LDFLAGS)

$(BUILD)/tsan/%: $(TESTDIR)/%.c $(T_OBJ) $(LIB_HDR)
	$(T_CC) $(T_CFLAGS) -o $@ $< $(T_OBJ) $(T_LDFLAGS)

$(BUILD)/linux/obj $(BUILD)/win/obj $(BUILD)/asan/obj $(BUILD)/tsan/obj:
	mkdir -p $@

# ---- running --------------------------------------------------------------------------------------
# Each suite runs from inside its build directory, so its scratch files never land in the source tree.
test: $(L_BIN)
	@set -e; for t in $(TESTS); do echo "=== linux: $$t"; (cd $(BUILD)/linux && ./$$t); done

win: $(W_BIN)

wine-test: $(W_BIN)
	@set -e; mkdir -p $(WINEPREFIX); \
	for t in $(TESTS); do echo "=== wine: $$t"; (cd $(BUILD)/win && $(WINE_ENV) wine ./$$t.exe); done

# The sanitizer variants run the suites in QUICK mode (ENGRAM_TEST_QUICK=1): every code path, a
# sixth of the measurement queries -- the full measurement runs natively and under Wine.
asan: $(A_BIN)
	@set -e; for t in $(TESTS); do echo "=== asan+ubsan: $$t"; \
	  (cd $(BUILD)/asan && ENGRAM_TEST_QUICK=1 ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
	   UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ./$$t); done

tsan: $(T_BIN)
	@set -e; for t in $(TESTS); do echo "=== tsan: $$t"; \
	  (cd $(BUILD)/tsan && ENGRAM_TEST_QUICK=1 TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1 ./$$t); done

# The planted-defect build the gate must see fail (R6). Never shipped, never part of `all`.
$(BUILD)/plant/test_core: $(TESTDIR)/test_core.c $(L_OBJ) $(LIB_HDR)
	mkdir -p $(BUILD)/plant
	$(L_CC) $(L_CFLAGS) -DENGRAM_GATE_PLANT -o $@ $< $(L_OBJ) $(L_LDFLAGS)

plant: $(BUILD)/plant/test_core

gate:
	@tools/gate.sh

clean:
	rm -rf $(BUILD)/linux $(BUILD)/win $(BUILD)/asan $(BUILD)/tsan $(BUILD)/plant
