use std::vec::Vec;

use super::neuron;

pub struct FeedFoward {
	inputs: Vec<neuron::Sigmoid>,
	hidden: Vec<neuron::Sigmoid>,
	outputs: Vec<neuron::Sigmoid>,
}

impl FeedFoward {
	pub fn new() -> FeedFoward {
        let mut feed = FeedFoward {
        	inputs: Vec::with_capacity(::INPUT_N as usize),
        	hidden: Vec::with_capacity(::HIDDEN_N as usize),
        	outputs: Vec::with_capacity(::OUTPUT_N as usize),
		};

		for _ in 0..::INPUT_N as usize
		{
			let sigmoid = neuron::Sigmoid::new(0);
			feed.inputs.push(sigmoid);
		}

		for _ in 0..::HIDDEN_N as usize
		{
			let mut sigmoid = neuron::Sigmoid::new(::INPUT_N as usize);
			sigmoid.randomize_weights();
			feed.hidden.push(sigmoid);
		}

		for i in 0..::OUTPUT_N as usize
		{
			let mut sigmoid = neuron::Sigmoid::new(::HIDDEN_N as usize);
			sigmoid.randomize_weights();
			feed.outputs.push(sigmoid);
		}

		println!("HELO WORLD");

        return feed;
    }

    pub fn run(&mut self, input_data: Vec<u8>) -> u8
    {
    	for i in 0..self.inputs.len()
    	{
    		self.inputs[i].set_output(input_data[i] as f64 / u8::max_value() as f64);
    	}

    	for i in 0..self.hidden.len()
    	{
    		self.hidden[i].output_from_inputs(&self.inputs);
    	}

    	for i in 0..self.outputs.len()
    	{
    		self.outputs[i].output_from_inputs(&self.hidden);
    	}

    	// Find largest output signal
    	let mut largest:u8 = 0;

    	//println!("Output length is: {:?}", self.outputs.len());
    	for i in 0..self.outputs.len()
    	{
    		//println!("Output value is: {:?}", self.outputs[i].out());
    		if self.outputs[i].out() > self.outputs[largest as usize].out()
    		{
    			largest = i as u8;
    		}
    	}

    	return largest;
    }

    pub fn teach(&mut self, input_data: Vec<u8>, supposed: u8)
    {
    	let current = self.run(input_data);

    	let current_cost = self.outputs[current as usize].out();
    	let supposed_cost = self.outputs[supposed as usize].out();

    	let error_amount = (supposed_cost - current_cost).abs();

    	/*println!("Current: {:?}", current);
    	println!("Supposed: {:?}", supposed);
    	if current == supposed
    	{
    		println!("MATCH");
    	}
    	else
    	{
    		println!("\n");
    	}*/

		for i in 0..self.outputs[current as usize].weights.len()
		{
			if current != supposed
			{
				self.outputs[current as usize].weights[i] -= 0.001 * error_amount*10.0;
				if self.outputs[current as usize].weights[i] < 0.0
				{
					self.outputs[current as usize].weights[i] = 0.0;
				}
			}
			else {
			    self.outputs[current as usize].weights[i] += 0.01 * error_amount*10.0;
			   	if self.outputs[current as usize].weights[i] > 1.0
				{
					self.outputs[current as usize].weights[i] = 1.0;
				}
			}
		}
    }
}