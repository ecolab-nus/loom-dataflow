pub struct Memory {
    pub ports: Vec<Port>,
    pub capacity: usize,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum Port {
    Read,
    Write,
    ReadWrite,
}
