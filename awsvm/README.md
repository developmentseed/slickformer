# Deploy GPU ML instance in AWS

## Setup

Follow the steps described in [this](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#authentication-and-configuration) to setup AWS authentication.

**When using GPUs, make sure to set appropriate Service Quotas in each region you want to deploy instances to.** You can view all service quotas in [here](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas). I you want to enable usage of P3 instances, you'll need to request an increase in "Running On-Demand P instances" quota (this is often set to 0 for vanilla AWS accounts).

Install the [terraform CLI](https://learn.hashicorp.com/tutorials/terraform/install-cli).

## Adapt `variables.tf` file

Navigate to the folder containing `main.tf`. Adapt the `variables.tf` file as needed, specifically the `instance-type` and the `location`.

## (Optional) Adapt `post-vm-setup-and-manual-steps.sh`

If you want to run your own setup script after the instance is done being created, edit the `post-vm-setup-and-manual-steps.sh` and run it from the git cloned repo.

## Deploy

Navigate to the folder containing the `main.tf` file. Run `terraform init`.

Check your deployment with `terraform plan`.

You can create your instance with `terraform apply`.

This will create a AWS EC2 instance, and save in your local machine a private ssh key (in `.ssh/`), and a series of `.vm-X` files containing identity information for your instance. **Do not delete or modify this files!**

## Currrently manual steps

Some steps are best accomplished manually, until we have a better workstation deployment system).

`git config --global -e` to edit your global config file. Copy your user configuration file from you local machine and paste into the vm's git configuration file.

Then, `gh auth login` and `gh repo clone developmentseed/slickformer`. Select the ssh option to add a new ssh key for the vm so you can push and access private repos

```bash
ubuntu@ip-172-31-27-67:~$ gh auth login
? What account do you want to log into? GitHub.com
? What is your preferred protocol for Git operations? SSH
? Generate a new SSH key to add to your GitHub account? Yes
? Enter a passphrase for your new SSH key (Optional)
? Title for your SSH key: GitHub CLI
? How would you like to authenticate GitHub CLI? Login with a web browser
ubuntu@ip-172-31-27-67:~$ gh repo clone developmentseed/slickformer
Cloning into 'slickformer'...
The authenticity of host 'github.com (140.82.121.3)' can't be established.
ECDSA key fingerprint is SHA256:p2QAMXNIC1TJYWeIOttrVc98/R1BUFWu3/LiyKgUfQM.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
```

For VSCode setup, go to Remote Explorer, select Remotes, select the wheel icon and edit the ssh config file. Add this block, adapting your absolute path to your pem key in the .ssh folder in awsvm.

 Note for WSL: I had to manually copy the .pem to another folder on the Windows partition so that VSCode could detect the .pem key. If on Mac just point to the project .ssh pem key created by terraform at `awsvm/.ssh/private_instance_aws.pem`.

```
Host slickformer-Dev
    HostName content from .vm-ip
    IdentityFile "~/slickformer/.ssh/private_instance_aws.pem"
    LocalForward 8888 localhost:8888
    LocalForward 6006 localhost:6006
    IdentitiesOnly yes
```


## `make` tools

You can now use the set of tools included in the `Makefile`. Adapt this file if needed in case you want to change the remote and local path to copy files into the instance.

- `make ssh`: connects to your instance in your shell. This also maps the port 8888 in the instance to your localhost, allowing you to serve a jupyter instance via this ssh tunnel (for instance by running `jupyter lab --allow-root`).
- `make start`, `make stop`, `make status` : Start, stop and check the status of your instance. **Important: if you are not using your instance, make sure to run `make stop` to avoid excessive costs! Don't worry, your instance state and files are safe.**
- `make syncup` and `make syncdown`: Copies files in your folder from and to your instance.

## Destroy

When you finish all work associated with this instance make sure to run `terraform destroy`. This will delete the ssh key in `.ssh` and all `.vm-X` files.

**Important: when you destroy your instance, all files and instance state are deleted with it so make sure to back them up to AWS or locally if needed!**
