Grocery Product Image Dataset
=============================

This dataset contains product images which were automatically captured at picking stations in warehouse.

* Image format: the images are in JPG format and have the dimensions of 1280 x 1024 pixels and RGB colour space.

* Image file naming convention: barcode_timestamp_m/p_milliseconds.jpg ( m/p means minus or plus )

Our camera keep capturing images every 25 milliseconds. When there is a barcode reading (a product is scanned),
the machine vision system saves four images, which are one image before the barcode reading (timestamp minus x milliseconds)
and three images after the barcode reading (timestamp plus y milliseconds). The timestamp specifies when it reads a barcode.

* Dataset structure:
   1) all the images for each barcode have been put together in a tarball file.
   2) unpack-dataset.sh can extract all the tarball files in one go.
   3) products.json file describes the details of each product.


Barcode           Images     Product Name
5022313312101       1000     Tropicana Orange & Mango Juice 850ml
5000169279335       1000     Waitrose Raw Jumbo King Prawns 180g
5000169322864       1000     Waitrose Duchy Organic 4 Chicken Wings 400g
5021638220412       1000     Rachel's Organic Fresh Whole Milk 1L
5000169036457       1000     Waitrose 1 Madagascan Fresh Vanilla Custard 500g
5000169181676       1000     Waitrose 1 Still Fresh Lemonade with Lemon Juice 1L
5000169102541       1000     Waitrose Duchy Organic Tenderstem Broccoli 200g
5000295145054       1000     Cathedral City Mature Cheddar 550g
5000169217429       1000     Waitrose 2 Wild Alaskan Keta Salmon Fillets 220g
3155250361825       1000     President Slightly Salted Spreadable 250g
5011069168919       1000     Fire & Smoke Flame Grilled Shaved Ham 100g
5055540026268       1000     Ocado Diced Lamb Leg 450g
5038862131087       1000     Innocent Coconut Water 500ml
5707361100053       1000     Anchor Spreadable 500g
5000169073018       1000     Waitrose 1 Ripe Papaya 2 per pack
3073781011456       1000     Leerdammer 8 Lightlife Slices 160g
5010171005204       1000     Country Life Unsalted British Butter 250g
5000169689707       1000     Cooks' Ingredients Mixed Thai Chilli 30g
5021638122730       1000     Rachel's Organic Low Fat Mango Yogurt 450g
5036589250579       1000     Little Yeo's Fruited Ltd Edition Yogurt 90g
5000169412053       1000     Waitrose Giant Couscous, Wheatberries & Sweet Potato 300g
5000169279793       1000     Essential Waitrose 2 Pork Chops 540g
4025500181871       1000     Muller Corner Bliss, Greek Style Whipped Yoghurt & Lemon Compote 4 x 110g
