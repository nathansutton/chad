import { Engine, helper } from "./lib";

export function process(x: number): number {
  return helper(x);
}

export function run(): number {
  return new Engine().process(process(1));
}
