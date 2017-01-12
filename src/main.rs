pub const  INPUT_N: i32 = 28*28;
pub const  LAYERS: i32 = 1;
pub const  HIDDEN_N: i32 = 30;
pub const  OUTPUT_N: i32 = 10;
pub const  IMAGE_SIZE: i32 = 28;

use std::io::Read;
use std::fs::File;
use std::fmt::Write;

mod network;

fn main() {
	// Create our network
    let mut feed_network = network::network::FeedFoward::new();

	//println!("{:?}", network::network::helloo());
    // Image labels
    let mut file = File::open("train-labels.idx1-ubyte").unwrap();

    let mut magic_number=[0u8;4];
    file.read(&mut magic_number).unwrap();
    println!("{:?}",magic_number);

    let mut number_of_images=[0u8;4];
    file.read(&mut number_of_images).unwrap();
    println!("{:?}",number_of_images);

    let mut labels=[0u8;60000];

    for i in 0..60000
    {
    	let mut label=[0u8;1];
    	file.read(&mut label).unwrap();

    	labels[i] = label[0];
    }

    // Image data
    file = File::open("train-images.idx3-ubyte").unwrap();

    magic_number=[0u8;4];
    file.read(&mut magic_number).unwrap();
    println!("{:?}",magic_number);

    number_of_images=[0u8;4];
    file.read(&mut number_of_images).unwrap();
    println!("{:?}",number_of_images);

    let mut number_of_rows=[0u8;4];
    file.read(&mut number_of_rows).unwrap();
    println!("{:?}",number_of_rows);

    let mut number_of_columns=[0u8;4];
    file.read(&mut number_of_columns).unwrap();
    println!("{:?}",number_of_columns);

    let mut color_row=[0u8;IMAGE_SIZE as usize];

    let mut image_data: Vec<u8> = Vec::with_capacity(IMAGE_SIZE as usize * IMAGE_SIZE as usize * 60000);
    file.read_to_end(&mut image_data).unwrap();

    println!("Image data length: {:?}",image_data.len());

    for i in 0..60000
    {
    	feed_network.teach(image_data[i*28*28..i*28*28 + 28*28].to_vec(), labels[i]);

    	if i%500 == 0
    	{
    		println!("Currently teaching image: {:?}", i);
    	}
    }

	println!("Number is: {:?}", feed_network.run(image_data[0..28*28].to_vec()));

    // Read test labels
    file = File::open("t10k-labels.idx1-ubyte").unwrap();

    magic_number=[0u8;4];
    file.read(&mut magic_number).unwrap();
    println!("{:?}",magic_number);

    number_of_images=[0u8;4];
    file.read(&mut number_of_images).unwrap();
    println!("{:?}",number_of_images);

    let mut test_labels=[0u8;10000];

    for i in 0..10000
    {
    	let mut label=[0u8;1];
    	file.read(&mut label).unwrap();

    	test_labels[i] = label[0];
    }

    // Read test images
    file = File::open("t10k-images.idx3-ubyte").unwrap();

    magic_number=[0u8;4];
    file.read(&mut magic_number).unwrap();
    println!("{:?}",magic_number);

    number_of_images=[0u8;4];
    file.read(&mut number_of_images).unwrap();
    println!("{:?}",number_of_images);

    number_of_rows=[0u8;4];
    file.read(&mut number_of_rows).unwrap();
    println!("{:?}",number_of_rows);

    number_of_columns=[0u8;4];
    file.read(&mut number_of_columns).unwrap();
    println!("{:?}",number_of_columns);

    //color_row=[0u8;28];

    image_data = Vec::with_capacity(IMAGE_SIZE as usize * IMAGE_SIZE as usize * 10000);
    file.read_to_end(&mut image_data).unwrap();

    println!("Image data length: {:?}",image_data.len());

    for i in 0..28
    {
    	let mut s = String::new();
		for &byte in image_data[i*28..i*28+28].iter() {
			write!(&mut s, "{:02x} ", byte).unwrap();
		}

    	//println!("{}", s);
    }

    // Well.. Let's try this shit out
    println!("Number is supposed to be: {:?}", test_labels[0]);
    println!("Number is: {:?}", feed_network.run(image_data[0..28*28].to_vec()));

    let mut wrong_answers:i32 = 0;
    let mut correct_answers:i32 = 0;

    for i in 0..10000
    {
    	let answer = feed_network.run(image_data[i*28*28..i*28*28 + 28*28].to_vec());
    	let supposed = test_labels[i];

    	if answer != supposed
    	{
    		wrong_answers += 1;
    	}
    	else {
    	    correct_answers += 1;
    	}

    	if i%500 == 0
    	{
    		println!("Currently studying image: {:?}", i);
    	}
    }
    println!("Wrong answers: {:?}", wrong_answers);
    println!("Correct answers: {:?}", correct_answers);
    println!("Correct rate: {:?}", correct_answers as f32/(wrong_answers as f32 + correct_answers as f32));
}
