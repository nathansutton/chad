#!/usr/bin/env bash

MODE=fast

process() {
  echo "lib:$1"
}

engine_process() {
  process "$1"
}

helper() {
  echo "$(($1 + 1))"
}
