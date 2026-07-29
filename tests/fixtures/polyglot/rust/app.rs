use crate::lib::{helper, Engine};

pub fn process(x: i32) -> i32 {
    helper(x)
}

pub fn run() -> i32 {
    Engine.process(process(1))
}
