variable "location" {
  type        = string
  description = "Location of the resources"
  default     = "eu-central-1"
}

variable "ami" {
  type        = string
  description = "AMI to launch VM"
  # Note that AMIs are region based
  # "ami-0050625d58fa27b6d" - aws deep learning AMI with ubuntu 18.04 in us-west-2
  # "ami-04505e74c0741db8d" - ubuntu 20.04 in us-west-2
  # "ami-0892d3c7ee96c0bf7" - ubuntu 20.04 in us-east-1
  # "ami-0c0d3776ef525d5dd" - amazon linux in eu-central-1 
  # "ami-0ad9796167d61b7ae" - ubuntu focal fossa in eu-central-1
  # If you change the region make sure to change the AMI as well
  # check all AMIs available here
  # https://console.aws.amazon.com/ec2/v2/home?AMICatalog
  # DevSeed DeepLearning AMIs:
  # - us-east-1: ami-09c1743af5e16c7d5
  # - us-west-2: ami-01e90c376d578a2f0
  default     = "ami-0ad9796167d61b7ae" 
}


variable "instance-type" {
  type        = string
  description = "Instance type to deploy"
  #default     = "p2.xlarge"
  default     = "p2.xlarge"
}