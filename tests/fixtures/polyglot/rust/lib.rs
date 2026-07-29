pub struct Engine;

impl Engine {
    pub fn process(&self, x: i32) -> i32 {
        x * 2
    }
}

pub fn helper(x: i32) -> i32 {
    x + 1
}
