#!/bin/sh

class0_dir_list=(
	E:/0-ML_database/handwriting/digits/train/0
)

class1_dir_list=(
	E:/0-ML_database/handwriting/digits/train/1
)

class2_dir_list=(
	E:/0-ML_database/handwriting/digits/train/2
)

class3_dir_list=(
	E:/0-ML_database/handwriting/digits/train/3
)

class4_dir_list=(
	E:/0-ML_database/handwriting/digits/train/4
)

class5_dir_list=(
	E:/0-ML_database/handwriting/digits/train/5
)

class6_dir_list=(
	E:/0-ML_database/handwriting/digits/train/6
)

class7_dir_list=(
	E:/0-ML_database/handwriting/digits/train/7
)

class8_dir_list=(
	E:/0-ML_database/handwriting/digits/train/8
)

class9_dir_list=(
	E:/0-ML_database/handwriting/digits/train/9
)

class0_list_file="train/class0.lst"
class1_list_file="train/class1.lst"
class2_list_file="train/class2.lst"
class3_list_file="train/class3.lst"
class4_list_file="train/class4.lst"
class5_list_file="train/class5.lst"
class6_list_file="train/class6.lst"
class7_list_file="train/class7.lst"
class8_list_file="train/class8.lst"
class9_list_file="train/class9.lst"

save_imagepath()
{
	local dir_list=$1
	local imagelist_file=$2
	
	rm -rf $imagelist_file
	touch $imagelist_file
	for dir in ${dir_list[*]}
	do
		find $dir -name "*" | grep -i -E "bmp|jpg|png" >> $imagelist_file
	done
}

rm -rf train
mkdir train

save_imagepath $class0_dir_list $class0_list_file
save_imagepath $class1_dir_list $class1_list_file
save_imagepath $class2_dir_list $class2_list_file
save_imagepath $class3_dir_list $class3_list_file
save_imagepath $class4_dir_list $class4_list_file
save_imagepath $class5_dir_list $class5_list_file
save_imagepath $class6_dir_list $class6_list_file
save_imagepath $class7_dir_list $class7_list_file
save_imagepath $class8_dir_list $class8_list_file
save_imagepath $class9_dir_list $class9_list_file
