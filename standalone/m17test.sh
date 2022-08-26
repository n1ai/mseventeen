#!/usr/bin/env bash

set -x

## Definitions ################################################################

beep() {
   # found a random file that makes a beep to signify a test is over
   play /usr/share/sounds/freedesktop/stereo/bell.oga >& /dev/null
}

banner() {
    printf "\n#######################################\n# $1\n#######################################\n\n"
}

export AUD="/usr/local/share/m17/apollo11_3210.aud"
export PLAY="play -q -b 16 -r 8000 -c1 -t s16 -"
export N1AI="./m17.py"
export WX9O_MOD="/usr/local/bin/m17-mod -S N0CAL"
export WX9O_DEM="/usr/local/bin/m17-demod"

## No encoding test case ############################################

banner "n1ai loopback with no encoding:"

${N1AI} -i ${AUD} -e none -S

beep

## Codec2 test cases ############################################

banner "n1ai loopback with codec2 encoding:"

${N1AI} -i ${AUD} -e codec2 -S

beep

banner "n1ai rx from file with only codec2 encoding:"

${N1AI} -f tx -i ${AUD} -o n1ai.c2 -e codec2

${N1AI} -f rx -i n1ai.c2 -S -e codec2

beep

banner "n1ai tx to pipe to rx with only codec2 encoding:"

${N1AI} -f tx -i ${AUD} -e codec2 | ${N1AI} -f rx -e codec2 -S

beep

## N1AI to/from N1AI sym test cases ###########################################

banner "n1ai loop from aud w/o speaker"

${N1AI} -i ${AUD} -o - | ${PLAY}

beep

banner "n1ai loop from aud with speaker:"

${N1AI} -i ${AUD} -S

beep

banner "n1ai tx from aud to pipe to n1ai rx w/o speaker:"

${N1AI} -f tx -i ${AUD} | ${N1AI} -f rx -o - | ${PLAY}

beep

banner "n1ai tx from aud to sym to pipe to n1ai rx with speaker:"

${N1AI} -f tx -i ${AUD} | ${N1AI} -f rx -S

beep

banner "n1ai tx from aud to sym:"

${N1AI} -f tx -i ${AUD} -o n1ai.sym

# no beep since we didn't play sound

banner "n1ai rx from sym with speaker:"

${N1AI} -f rx -i n1ai.sym -S

beep

banner "n1ai rx from sym to aud then play:"

${N1AI} -f rx -i n1ai.sym -o n1ai.aud; ${PLAY} < n1ai.aud

beep

## N1AI to/from WX9O sym test cases ############################################

banner "wx9o tx from aud to sym to pipe to wx90 demod (sanity):"

${WX9O_MOD} --sym < ${AUD} | ${WX9O_DEM} --sym -l | ${PLAY}

beep

banner "n1ai tx from aud to sym to pipe to wx90 demod (sanity):"

${N1AI} -f tx -i ${AUD} | ${N1AI} -f rx -S

# without pipe:
# ${N1AI} -i ${AUD} -S

beep

banner "n1ai tx from aud to sym to pipe to wx90 demod:"

${N1AI} -f tx -i ${AUD} | ${WX9O_DEM} --sym -l | ${PLAY}

beep

banner "wx9o tx from aud to sym to pipe to n1ai demod:"

${WX9O_MOD} --sym < ${AUD} | ${N1AI} -f rx -S

beep

# ${WX9O_MOD} --sym < ${AUD} > wx9o.sym
# ${N1AI} -f rx -i wx9o.sym -S
# ${N1AI} -i ${AUD} -S

## N1AI to/from N1AI bin test cases ############################################

banner "n1ai loop from aud w/o speaker"

# TODO: verify this used bin mode!
${N1AI} -i ${AUD} -b -o - | ${PLAY}

beep

banner "n1ai loop from aud with speaker:"

# TODO: verify this used bin mode!
${N1AI} -i ${AUD} -b -S

beep

banner "n1ai tx from aud to pipe to n1ai rx w/o speaker:"

# TODO: verify this used bin mode!
${N1AI} -f tx -i ${AUD} -b | ${N1AI} -f rx -b -o - | ${PLAY}

beep

banner "n1ai tx from aud to bin to pipe to n1ai rx with speaker:"

${N1AI} -f tx -i ${AUD} -b | ${N1AI} -f rx -b -S

beep

banner "n1ai tx from aud to bin:"

${N1AI} -f tx -i ${AUD} -b -o n1ai.bin

# no beep since we didn't play sound

banner "n1ai rx from bin with speaker:"

${N1AI} -f rx -i n1ai.bin -b -S

beep

banner "n1ai rx from bin to aud then play:"

${N1AI} -f rx -i n1ai.bin -b -o n1ai.aud; ${PLAY} < n1ai.aud

beep

## N1AI to/from WX9O bin test cases ############################################

banner "wx9o tx from aud to bin to pipe to wx90 demod (sanity):"

${WX9O_MOD} --bin < ${AUD} | ${WX9O_DEM} --bin -l | ${PLAY}

beep

banner "n1ai tx from aud to sym to pipe to wx90 demod (sanity):"

${N1AI} -f tx -i ${AUD} -b | ${N1AI} -f rx -b | ${PLAY}

beep

# without pipe:

banner "n1ai tx from aud to sym to pipe to wx90 demod (sanity):"

${N1AI} -i ${AUD} -b -S

beep

banner "n1ai tx from aud to sym to pipe to wx90 demod:"

${N1AI} -f tx -i ${AUD} -b | ${WX9O_DEM} --bin -l | ${PLAY}

beep

banner "wx9o tx from aud to sym to pipe to n1ai demod:"

${WX9O_MOD} --bin < ${AUD} | ${N1AI} -f rx -b -S

beep

# ${WX9O_MOD} --sym < ${AUD} > wx9o.sym
# ${N1AI} -f rx -i wx9o.sym -S
# ${N1AI} -i ${AUD} -S

# cleanup

rm -f n1ai.bin n1ai.sym n1ai.aud n1ai.c2

exit  # bye!

