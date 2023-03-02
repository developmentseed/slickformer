terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.71"
    }
    tls = {
      source = "hashicorp/tls"
      version = "3.2.0"
    }
    local = {
      source = "hashicorp/local"
      version = "2.1.0"
    }
  }

  required_version = ">= 0.14.9"
}

provider "aws" {
  region  = "${var.location}"
}

locals {
  name = "ml-jupyter-slickformer" #testing to see if fixing name results in fewer destroyed resources when isntance type changed
}
