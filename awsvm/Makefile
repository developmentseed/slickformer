# Settings
REMOTE_DIR=~/work
LOCAL_DIR=../

##### You should not have to touch stuff below this line

PATH_SSH_KEY=.ssh/private_instance_aws.pem
PATH_VM_ID=.vm-id
PATH_VM_IP=.vm-ip
PATH_VM_NAME=.vm-name
PATH_VM_REGION=.vm-region

VM_ID=$(shell cat ${PATH_VM_ID}) # AWS ARN
VM_CONN=$(shell cat ${PATH_VM_IP}) # user@IP
VM_NAME=$(shell cat ${PATH_VM_NAME}) # name
VM_REGION=$(shell cat ${PATH_VM_REGION}) # region

mkfile_path=$(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR=$(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

# sync code
syncup:
	ssh -i ${PATH_SSH_KEY} -o IdentitiesOnly=yes ${VM_CONN} "mkdir -p ${REMOTE_DIR}"
	rsync -e "ssh -i ${PATH_SSH_KEY} -o IdentitiesOnly=yes" -avz --exclude ".git/" --exclude "awsvm/" --exclude-from="${LOCAL_DIR}.gitignore" ${LOCAL_DIR} $(shell echo ${VM_CONN}:${REMOTE_DIR}/ | tr -d '[:space:]')
syncdown:
	rsync -e "ssh -i ${PATH_SSH_KEY} -o IdentitiesOnly=yes" -avz --exclude ".git/" --exclude "awsvm/" --exclude-from="${LOCAL_DIR}.gitignore" $(shell echo ${VM_CONN}:${REMOTE_DIR}/| tr -d '[:space:]') ${LOCAL_DIR}

# start/stop instance
stop:
	aws ec2 stop-instances --instance-ids ${VM_ID} --region ${VM_REGION}
start:
	aws ec2 start-instances --instance-ids ${VM_ID} --region ${VM_REGION}
status:
	aws ec2 describe-instance-status --instance-ids ${VM_ID} --region ${VM_REGION}

# ssh into machine with jupyter lab
ssh:
	ssh -i ${PATH_SSH_KEY} -L 8888:localhost:8888 -L 6006:localhost:6006 -o IdentitiesOnly=yes ${VM_CONN}
