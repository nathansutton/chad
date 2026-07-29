import { Engine, helper } from "./lib.js";

export function process(x) {
  return helper(x);
}

export function run() {
  return new Engine().process(process(1));
}
