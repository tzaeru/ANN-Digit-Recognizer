use std::vec::Vec;

extern crate rand;
use network::neuron::rand::Rng;

pub fn hello() -> String {
    "Hello!".to_string()
}

pub struct Sigmoid {
	pub weights: Vec<f64>,
	cached_output: f64
}

impl Sigmoid {
	pub fn new(s: usize) -> Sigmoid {
        let mut sigmoid = Sigmoid { 
            weights: Vec::with_capacity(s), 
            cached_output: 0.0,
        };

        for i in 0..s
    	{
    		sigmoid.weights.push(1.0);
    	}

        return sigmoid;
    }

    pub fn randomize_weights(&mut self) {
    	let mut rng = rand::thread_rng();

    	for i in 0..self.weights.len()
    	{
    		self.weights[i] = rand::random::<f64>();
    	}
    }

	pub fn output_from_inputs(&mut self, inputs: &Vec<Sigmoid>) {
		let mut sum = 0.0;

        for i in 0..inputs.len() as usize
        {
        	sum += inputs[i].out()*self.weights[i];
        }
        self.cached_output = sum;
	}

	pub fn set_output(&mut self, output: f64)
	{
		self.cached_output = output;
	}

    pub fn out(&self) -> f64 {
    	self.cached_output
    }
}
