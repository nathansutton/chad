#!/usr/bin/env bash

source ./lib.sh

process() {
  helper "$1"
}

run() {
  engine_process "$(process 1)"
}
