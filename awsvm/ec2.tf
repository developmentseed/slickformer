resource "aws_security_group" "jupyter-sg" {
  tags = {
    Name = "${local.name}-jupyter-sg"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group_rule" "port888" {
  security_group_id = aws_security_group.jupyter-sg.id
  type              = "ingress"
  from_port         = 0
  to_port           = 8888
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "tls_private_key" "key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}
resource "aws_key_pair" "generated_key" {
  key_name   = "key-${uuid()}"
  public_key = tls_private_key.key.public_key_openssh
}
resource "local_file" "pem" {
  filename        = ".ssh/private_instance_aws.pem"
  content         = tls_private_key.key.private_key_pem
  file_permission = "600"
}

resource "aws_iam_instance_profile" "s3_profile" {
  name = "S3Profile-${local.name}"
  role = aws_iam_role.role.name
}

resource "aws_iam_role" "role" {
  name = "S3Role-${local.name}"
  path = "/"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
    ]
  })

  inline_policy {
    name = "s3_access"

    policy = jsonencode(
      {
        "Version" : "2012-10-17",
        "Statement" : [
          {
            "Effect" : "Allow",
            "Action" : [
              "s3:GetObject",
              "s3:ListBucket",
              "s3:PutObject"
            ],
            "Resource" : [
              "arn:aws:s3:::*",
              "arn:aws:s3:::*/*"
            ]
          }
        ]
    })
  }
}

resource "aws_instance" "ml_instance" {

  ami                         = var.ami
  instance_type               = var.instance-type
  availability_zone           = data.aws_availability_zones.available.names[0]
  vpc_security_group_ids      = ["${aws_security_group.jupyter-sg.id}"]
  associate_public_ip_address = true
  key_name                    = aws_key_pair.generated_key.key_name
  iam_instance_profile        = aws_iam_instance_profile.s3_profile.name

  tags = {
    Name = "${local.name}"
  }

  root_block_device {
    volume_size = 50
  }
  provisioner "remote-exec" {
    script = "vm-setup.sh"
    connection {
      type        = "ssh"
      host        = aws_instance.ml_instance.public_ip
      user        = "ubuntu" #ubuntu for ubuntu image
      private_key = local_file.pem.content
    }
  }

}

data "aws_availability_zones" "available" {
  state = "available"
}

# assign a constant IP that we can reach
resource "aws_eip" "ml_instance_ip" {
  instance = aws_instance.ml_instance.id
  vpc      = true

  tags = {
    Name = "${local.name}-ip"
  }
}

data "aws_region" "current" {}
resource "local_file" "vm_id" {
  filename = ".vm-id"
  content  = split("/", "${aws_instance.ml_instance.arn}")[1]
}

resource "local_file" "vm_name" {
  filename = ".vm-name"
  content  = aws_instance.ml_instance.tags["Name"]
}

resource "local_file" "vm_ip" {
  filename = ".vm-ip"
  content  = "ubuntu@${aws_eip.ml_instance_ip.public_ip}"
}

resource "local_file" "region" {
  filename = ".vm-region"
  content  = data.aws_region.current.name
}