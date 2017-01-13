pub const  INPUT_N: i32 = 28*28;
pub const  LAYERS: i32 = 1;
pub const  HIDDEN_N: i32 = 25;
pub const  OUTPUT_N: i32 = 10;
pub const  IMAGE_SIZE: i32 = 28;

pub const  TEACH_SIZE: i32 = 60000;
pub const  TEST_SIZE: i32 = 10000;

pub const  EPOCHS: i32 = 50;

use std::io::Read;
use std::fs::File;
use std::fmt::Write;

mod network;

fn get_image_data(file_name:&str, amount:usize) -> Vec<u8>
{
	// Image data
    let mut file = File::open(file_name).unwrap();

    let mut magic_number=[0u8;4];
    file.read(&mut magic_number).unwrap();
    println!("{:?}",magic_number);

    let mut number_of_images=[0u8;4];
    file.read(&mut number_of_images).unwrap();
    println!("{:?}",number_of_images);

    let mut number_of_rows=[0u8;4];
    file.read(&mut number_of_rows).unwrap();
    println!("{:?}",number_of_rows);

    let mut number_of_columns=[0u8;4];
    file.read(&mut number_of_columns).unwrap();
    println!("{:?}",number_of_columns);

    let mut color_row=[0u8;IMAGE_SIZE as usize];

    let mut image_data: Vec<u8> = Vec::with_capacity(IMAGE_SIZE as usize * IMAGE_SIZE as usize * amount);
    file.read_to_end(&mut image_data).unwrap();

    println!("Image data length: {:?}",image_data.len());

    return image_data;
}

fn get_image_labels(file_name:&str, amount:usize) -> Vec<u8>
{
    let mut file = File::open(file_name).unwrap();

    let mut magic_number=[0u8;4];
    file.read(&mut magic_number).unwrap();
    println!("{:?}",magic_number);

    let mut number_of_images=[0u8;4];
    file.read(&mut number_of_images).unwrap();
    println!("{:?}",number_of_images);

    let mut labels:Vec<u8> = Vec::with_capacity(amount);

    for i in 0..amount
    {
    	let mut label=[0u8;1];
    	file.read(&mut label).unwrap();

    	labels.push(label[0]);
    }

    return labels;
}

fn main() {
	// Create our network
    let mut feed_network = network::network::FeedFoward::new();

	//println!("{:?}", network::network::helloo());
    // Teaching image labels
    let mut teach_labels = get_image_labels("train-labels.idx1-ubyte", 60000);
    // Teaching image data
    let mut teach_image_data = get_image_data("train-images.idx3-ubyte", 60000);

	// Read test labels
	let test_labels = get_image_labels("t10k-labels.idx1-ubyte", 10000);
	// Read test images
	let test_image_data = get_image_data("t10k-images.idx3-ubyte", 10000);

    for epoch in 0..EPOCHS
    {
		for i in 0..TEACH_SIZE as usize
		{
			feed_network.teach(teach_image_data[i*28*28..i*28*28 + 28*28].to_vec(), teach_labels[i]);

			if i%5000 == 0
			{
				//println!("Currently teaching image: {:?}", i);
			}
		}

		// Well.. Let's try this shit out
		//println!("Number is supposed to be: {:?}", test_labels[0]);
		//println!("Number is: {:?}", feed_network.run(test_image_data[0..28*28].to_vec()));

		let mut wrong_answers:i32 = 0;
		let mut correct_answers:i32 = 0;

		for i in 0..TEST_SIZE as usize
		{
			let answer = feed_network.run(test_image_data[i*28*28..i*28*28 + 28*28].to_vec());
			let supposed = test_labels[i];

			if answer != supposed
			{
				wrong_answers += 1;
			}
			else {
			    correct_answers += 1;
			}

			if i%1000 == 0
			{
				//println!("Currently studying image: {:?}", i);
			}
		}
		println!("Epoch is: {:?}", epoch);
		println!("Wrong answers: {:?}", wrong_answers);
		println!("Correct answers: {:?}", correct_answers);
		println!("Correct rate: {:?}", correct_answers as f32/(wrong_answers as f32 + correct_answers as f32));
	}
}
