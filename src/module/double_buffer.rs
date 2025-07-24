use crate::component::memory::{Memory, Port};

pub struct DoubleBuffer {
    pub tok_size: usize,
}

impl DoubleBuffer {
    pub fn new(buf_size: usize) -> Self {
        Self { tok_size: buf_size }
    }

    pub fn concretize(&self, memory: &Memory) -> Option<Memory> {
        // The double buffer module requires the memory to be at least twice the size of the token size
        // The double buffer module requires at least two ports, one reading port and one writing port
        if memory.capacity >= self.tok_size * 2 && memory.ports.len() >= 2 {
            // Only support memory with at most one write port
            assert!(
                memory.ports.iter().filter(|&p| *p == Port::Write).count() <= 1,
                "Double buffer module only supports memory with at most one write port"
            );
            let mut ports = memory.ports.clone();
            // needs at least one write more
            if memory.ports.iter().filter(|&p| *p == Port::Write).count() == 0 {
                return None;
            } else {
                // Remove the write port
                ports.remove(ports.iter().position(|&p| p == Port::Write).unwrap());
                // Remove another read port
                ports.remove(ports.iter().position(|&p| p == Port::Read).unwrap());
                Some(Memory {
                    capacity: memory.capacity - self.tok_size * 2,
                    ports,
                })
            }
        } else {
            None
        }
    }
}
