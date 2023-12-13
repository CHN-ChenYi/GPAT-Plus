#!/bin/bash
# Run executables in folder named '*-bin' in the INPUT_PATH and generate a log with execution times

# Gets the location of the script
function getScriptLocation {
    SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
        DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
        SOURCE="$(readlink "$SOURCE")"
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
    done
        DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
        echo $DIR
}

function Help()
{
   echo "Run executables in folder named '*-bin' in the INPUT_DIR and generate a log with execution times"
   echo "Output logs are generated inside the output dir"
   echo
   echo "Syntax: $(basename $0) -D DATASET_SIZE INPUT_DIR OUTPUT_DIR"
   echo "Options:"
   echo -e "\t-D [SIZE]: Specify size of dataset: MINI, SMALL, MEDUIM, LARGE, EXTRALARGE"
   echo -e "\t-h: Print this Help."
   echo -e "\t-p: Collect perf event counters."
   echo -e "\t-v: Verbose mode."
   echo -e "\t-r [NUMBER]: Specify repetition number (overrides defaults)."
   echo
   echo -e "Repetition number of executions by default are:"
   echo -e "\tMINI:       10000"
   echo -e "\tSMALL:      4000"
   echo -e "\tMEDIUM:     1000"
   echo -e "\tLARGE:      50"
   echo -e "\tEXTRALARGE: 10"
}

function sourceSpecFile() {
  # Gets this script path.
  scriptPath=$(getScriptLocation)
  specFilePath="$scriptPath/../spec.file"
  # Checks the spec.file
  if [ ! -f $specFilePath ]
  then
      echo "Please create spec.file"
      Help
      exit 2
  fi
  source $specFilePath
}

function checkSpecs() {
  if [[ -z $CORES ]]; then
    echo "Number of logical processors not defined"
    exit 1
  fi
}

function CheckPerfParanoid()
{
  paranoid=$(cat '/proc/sys/kernel/perf_event_paranoid')
  if [ "$paranoid" -gt "1" ]; then
    echo "Please set kernel event paranoid to a least 1 to be able to use perf"
    echo "Use the following command:"
    echo "    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'"
    exit 1
  fi
}

RED="\033[0;91m"
GREEN="\033[0;92m"
YEL="\033[0;93m"
CLR="\033[0m"

function echoRed() {
  echo -e "$RED$1$CLR"
}

function echoGreen() {
  echo -e "$GREEN$1$CLR"
}

function echoYellow() {
  echo -e "$YEL$1$CLR"
}

DATASET_SIZE=""
collect_perf="false"
CUSTOM_REPEATS=""

while getopts ":hvpD:r:" option; do
  case $option in
    h)
      Help
      exit 0
      ;;
    v)
      set -x
      ;;
    p)
      collect_perf="true"
      ;;
    D)
      DATASET_SIZE=${OPTARG}
      ;;
    r)
      CUSTOM_REPEATS=${OPTARG}
     ;;
    \?)
      echo "Invalid option." >&2
      Help
      exit 1
      ;;
  esac
done

# Gets the last argument and checks if it is a directory.
OUTPUT_DIR=${BASH_ARGV[0]}
if [[ ! -d $OUTPUT_DIR ]] || [[ -z $OUTPUT_DIR ]]; then
    echo "ERROR: Output path is empty or isn't a directory."
    Help
    exit 1
fi
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

# Gets the second last argument and checks if it is a directory.
INPUT_DIR=${BASH_ARGV[1]}
if [[ ! -d $INPUT_DIR ]] || [[ -z $INPUT_DIR ]]; then
    echo "ERROR: Input path is empty or isn't a directory."
    Help
    exit 1
fi
INPUT_DIR=$(realpath $INPUT_DIR)

# Check dataset size
case $DATASET_SIZE in
  (MINI|SMALL|MEDIUM|LARGE|EXTRALARGE)
  ;;
  *)
    echo "Please choose a valid dataset size selection"
    echo "Use MINI, SMALL, MEDIUM, LARGE, or EXTRALARGE"
    exit 1
  ;;
esac

# Source spec
sourceSpecFile
checkSpecs

# Check if perf will work
if [[ "$collect_perf" == "true" ]]; then
  CheckPerfParanoid
fi

# Check if output dir is clean
if [ ! -n "$(find $OUTPUT_DIR -prune -empty)" ]; then
    echo "ERROR: Output log dir is not empty."
    echo "Please remove existing logs from $OUTPUT_DIR"
    exit 1
fi

cd $INPUT_DIR

case $DATASET_SIZE in
  MINI)
    REPEATS=10000
  ;;
  SMALL)
    REPEATS=4000
  ;;
  MEDIUM)
    REPEATS=1000
  ;;
  LARGE)
    REPEATS=50
  ;;
  EXTRALARGE)
    REPEATS=10
  ;;
  *)
    echo "Not a valid tile size selection. Use MINI, SMALL, MEDIUM, LARGE, EXTRALARGE."
    exit 1
  ;;
esac

if [[ ! -z $CUSTOM_REPEATS ]]; then
  REPEATS=$CUSTOM_REPEATS
fi

for binaries in $(find $INPUT_DIR -type d -name "*-bin" | sort); do
  echoGreen "\nRUNNING: $(basename $binaries)"

  output_log_name="$(basename ${binaries%-bin}).log"
  output_log_path="$OUTPUT_DIR/$output_log_name"

  for run in $(seq $REPEATS); do
    echo "Iteration $run of $REPEATS -------------------------------"

    for i in $(find $binaries -name "*.exe" | shuf); do
      echo Running "$(basename $i)" on all cores | tee -a $output_log_path

      for core in $(seq 0 $((CORES-1))); do
        if [[ "$collect_perf" == "true" ]]; then
          taskset -c $core perf stat -e '{cycles,instructions},{mem_load_retired.l1_miss,mem_load_retired.l2_miss,mem_load_retired.l3_miss},{dtlb_load_misses.stlb_hit,dtlb_load_misses.miss_causes_a_walk}' $i 2>&1 | tee -a $output_log_path &
        else
          taskset -c $core $i 2>&1 | tee -a $output_log_path &
        fi
      done

      wait # Wait for all instances to complete
    done
    echo -e "\n"
  done
done

