pipeline {
    agent any
    
    environment {
        POST_BUILD_URL = 'https://7a44b00944e0.ngrok.io/api/v1/event/'
        BUILD_BEGIN_MESSAGE = ''
        BUILD_SUCCESS_MESSAGE = ''
        BUILD_FAILURE_MESSAGE = ''
        
    }
    
    stages {
        stage('Create Instance and Run Ansible Playbook') {
            steps {
                sh 'mkdir -p instance-config'
                dir ('instance-config'){
                    withAWS(credentials: 'aws-devops', region: 'us-east-2') {
                            sh '''
                            sudo rm -rf *
                            sudo aws s3 sync s3://instance-config-model-build/ ./
                            '''
                    }
                }
                withCredentials([file(credentialsId: 'ansible-key-2-pvt', variable: 'SSH_KEY'),
                file(credentialsId: 'ansible-ec2-key', variable: 'EC2_PEM')]) {
                    sh "sudo cp $SSH_KEY ./"
                    sh "sudo chmod 600 ansible-key-2"
                    sh "sudo cp $EC2_PEM ./"
                    sh "sudo chmod 600 ansible-ec2.pem"
                }
                script{
                config = readYaml (file: 'instance-config/instance-config.yml')
                parseGeneralConfig(config)
                notifyPipelineBeginning()
                def models = config.models
                for ( model in models ) {
                    if(model.enabled == true) {
                        parseModelBuildConfig(model)
                        getModelName()
                        notifyBuildBeginStatus()
                        try {
                            if (env.CLOUD_PROVIDER == "aws") {
                                echo "building model ins aws"
                                env.SSH_KEY_FILE = 'ansible-ec2.pem'
                                // buildModelAws()
                            } else if (env.CLOUD_PROVIDER == "gcp") {
                                env.SSH_KEY_FILE = 'ansible-key-2'
                                if ((model.gcpGpuModel != null)&&(model.gcpGpuCount != null)) {
                                    echo "building model in gcp with gpu"
                                    // buildModelGcpWithGpu()
                                } else {
                                    echo "building model in gcp without gpu"
                                    buildModelGcpWithoutGpu()
                                }
                            } else {
                                error "cloud provider cannot be recognised"
                            }
                        } catch (Exception e) {
                            notifyBuildFailureStatus()
                            throw e
                        }
                        try {
                            downloadAnsibleConfig()
                            updateAnsibleConfig()
                            checkOpenPortSSH()
                            runAnsiblePlaybook()
                            notifyBuildSuccessStatus()
                            // terminateInstance()
                        } catch (Exception e) {
                            // terminateInstance()
                            notifyBuildFailureStatus()
                            throw e
                        }
                    }
                    else {
                        echo "skipping current scipt" 
                    }
                }
                }
                // notifyPipelineEnd()
            }
        }
    }
    
    post {
        success {
            notifyPipelineSuccess()
        }
        failure {
            notifyPipelineFailure()
        }
    }
}

def parseGeneralConfig(config) {
    if (config.defaultCloudProvider == null) { 
        echo "default cloud provider not specified"
        error "default cloud provider not specified"
    } else { 
        env.DEFAULT_CLOUD_PROVIDER = config.defaultCloudProvider.toString()
    }
    if (config.defaultInstanceType == null) { 
        echo "default isntance type not specified"
        error "default isntance type not specified"
    } else { 
        env.DFAULT_INSTANCE_TYPE = config.defaultInstanceType.toString()
    }
    if (config.defaultCpuAmi == null) { 
        echo "default cpu ami not specified"
        error "default cpu instance ami not specified"
    } else { 
        env.DEFAULT_CPU_AMI = config.defaultCpuAmi.toString()
    }
    if (config.defaultGpuAmi == null) { 
        echo "default gpu ami not specified"
        error "default gpu instance ami not specified"
    } else { 
        env.DEFAULT_GPU_AMI = config.defaultGpuAmi.toString()
    }
    if (config.defaultAwsRegion == null) { 
        echo "default aws region not specified"
        error "default aws region not specified"
    } else { 
        env.DEFAULT_AWS_REGION = config.defaultAwsRegion.toString()
    }
    if (config.defaultGcpRegion == null) { 
        echo "default gcp region not specified"
        error "default gcp region not specified"
    } else { 
        env.DEFAULT_GCP_REGION = config.defaultGcpRegion.toString()
    }
    if (config.defaultGcpZone == null) { 
        echo "default gcp zone not specified"
        error "default gcp zone not specified"
    } else { 
        env.DEFAULT_GCP_ZONE = config.defaultGcpZone.toString()
    }
    if (config.usecase == null) { 
        echo "no use case id specified"
        error "no use case id specified"
    } else { 
        env.USECASE = config.usecase.toString()
    }
    if (config.application_id == null) { 
        echo "no application id specified"
        error "no application id specified"
    } else { 
        env.APPLICATION_ID = config.application_id.toString()
    }
    if (config.cloudProvider == null) { 
        echo "default cloud provider selected"
        env.CLOUD_PROVIDER = env.DEFAULT_CLOUD_PROVIDER
    } else { 
        env.CLOUD_PROVIDER = config.cloudProvider.toString()
    }
}

def parseModelBuildConfig (model) {
    if (model.instanceType == null) {
        echo 'default instance selected'
        env.INSTANCE_TYPE = env.DFAULT_INSTANCE_TYPE
        echo "${INSTANCE_TYPE}"
    } else {
        env.INSTANCE_TYPE = model.instanceType.toString()
        echo "${INSTANCE_TYPE}"
    }
    if (env.CLOUD_PROVIDER == "aws") {
        if(env.INSTANCE_TYPE[0..1].matches("g2|g3|g4|P2|P3|F1|Inf1")) {
            echo 'default gpu ami selected'
            env.AMI = env.DEFAULT_GPU_AMI
            echo "${AMI}"
        } else{
            echo 'default cpu ami selected'
            env.AMI = env.DEFAULT_CPU_AMI
            echo "${AMI}"
        }
        if (model.region == null) {
            echo 'default aws eegion selected'
            env.REGION = env.DEFAULT_AWS_REGION
            echo "${REGION}"
        } else {
            env.REGION = model.region.toString()
            echo "${REGION}"
        }
    } else if (env.CLOUD_PROVIDER == "gcp") {
        if (model.region == null) {
            echo 'default gcp region selected'
            env.REGION = env.DEFAULT_GCP_REGION
            echo "${REGION}"
        } else {
            env.REGION = model.region.toString()
            echo "${REGION}"
        }
        if (model.gcpZone == null) {
            echo 'default gcp zone selected'
            env.GCP_ZONE = env.DEFAULT_GCP_ZONE
            echo "${GCP_ZONE}"
        } else {
            env.GCP_ZONE = model.gcpZone.toString()
            echo "${GCP_ZONE}"
        }
        if ((model.gcpGpuModel != null)&&(model.gcpGpuCount != null)) {
            if(env.INSTANCE_TYPE[0..1].matches("n1")) {
                if(env.INSTANCE_TYPE[0..11].matches("n1-standard")) {
                    env.GCP_GPU_MODEL = model.gcpGpuModel.toString()
                    env.GCP_GPU_COUNT = model.gcpGpuCount.toString()
                    echo "gpu model and gpu count is set"
                } else {
                    error "GPUs are currently only supported with general-purpose N1 machine types. Specified machine type is not N1-standard"
                }  
            } else {
                error "GPUs are currently only supported with general-purpose N1 machine types. Specified machine type is not N1"
            }
        } else if ((model.gcpGpuModel != null)&&(model.gcpGpuCount == null)) {
            error "gpu count not specified"
        } else if ((model.gcpGpuModel == null)&&(model.gcpGpuCount != null)) {
            error "gpu model not specified"
        } else {
            echo "gpu not specified. proceeding with cpu only machine"
        }
    }
    if (model.scriptName == null) {
        echo 'script not specified. exiting'
        error "script not specified. job failed"
    } else { 
        env.SCRIPT_NAME = model.scriptName.toString()
        echo "${SCRIPT_NAME}"
        env.INSTANCE_NAME = env.SCRIPT_NAME.minus(".py").replaceAll("_","")+"-"+BUILD_NUMBER
        echo "${INSTANCE_NAME}"
    }
}

def buildModelAws () {
    withAWS(credentials: 'aws-devops', region: 'us-east-1') {
        script {
            sh label: '', script: '''
                ID=`aws ec2 run-instances --image-id $AMI --count 1 --instance-type $INSTANCE_TYPE \
                 --key-name ansible-ec2 --region $REGION  \
                 --iam-instance-profile Arn="arn:aws:iam::139189650355:instance-profile/s3access-EC2" \
                 --query 'Instances[0].InstanceId' --output text`
                aws ec2 wait instance-status-ok --instance-ids $ID
                PUBLIC_DNS_NAME=`aws ec2 describe-instances --region $REGION --filters Name=instance-id,Values=$ID --query Reservations[*].Instances[*].[PublicDnsName] --output text`
                '''
            env.IP = sh label: '', returnStdout: true, script: 'PUBLIC_DNS_NAME=`aws ec2 describe-instances --region $REGION --filters Name=instance-id,Values=$ID --query Reservations[*].Instances[*].[PublicDnsName] --output text`'
            env.IP = "${env.IP}".trim();
            env.INSTANCE_ID=sh label: '', returnStdout: true, script: '''
                aws ec2 describe-instances \
                    --filters Name=ip-address,Values=${IP} \
                    --query Reservations[*].Instances[*].[InstanceId] \
                    --output text
            '''
        }
        echo "IP = ${env.IP}";
        echo "INSTANCE_ID = ${INSTANCE_ID}"
    }
}

def buildModelGcpWithoutGpu () {
    withCredentials([file(credentialsId: 'gce-key', variable: 'GCE_KEY')]) {
        sh label: '', script: '''
            gcloud auth activate-service-account --key-file=${GCE_KEY}
            gcloud compute instances create $INSTANCE_NAME \
            --zone=${REGION}-${GCP_ZONE} \
            --image-family=ubuntu-1804-lts \
            --image-project ubuntu-os-cloud \
            --machine-type=$INSTANCE_TYPE \
            --boot-disk-size=10GB
            IP=`gcloud compute instances describe --zone=${REGION}-${GCP_ZONE} $INSTANCE_NAME \
                --format='get(networkInterfaces[0].accessConfigs[0].natIP)'`
        '''
        script {
            env.IP = sh label: '', returnStdout: true, script: 'gcloud compute instances describe --zone=${REGION}-${GCP_ZONE} $INSTANCE_NAME --format=\'get(networkInterfaces[0].accessConfigs[0].natIP)\''
            env.IP = "${env.IP}".trim();
        }
        echo "IP = ${env.IP}";
    }
}

def buildModelGcpWithGpu () {
    withCredentials([file(credentialsId: 'gce-key', variable: 'GCE_KEY')]) {
                sh label: '', script: '''
                    gcloud auth activate-service-account --key-file=${GCE_KEY}
                    gcloud compute instances create $INSTANCE_NAME \
                    --zone=${REGION}-${GCP_ZONE} \
                    --image-family=ubuntu-1804-lts \
                    --image-project ubuntu-os-cloud \
                    --machine-type=$INSTANCE_TYPE \
                    --boot-disk-size=10GB \
                    --accelerator type=$GCP_GPU_MODEL,count=$GCP_GPU_COUNT
                    gcloud compute instances add-metadata $INSTANCE_NAME --zone=${REGION}-${GCP_ZONE} --metadata-from-file ssh-keys=gcp_ssh_key.pub
                    sudo chmod 600 gcp_ssh_key
                    IP=`gcloud compute instances describe --zone=${REGION}-${GCP_ZONE} $INSTANCE_NAME \
                        --format='get(networkInterfaces[0].accessConfigs[0].natIP)'`
                    ssh -i gcp_ssh_key gcp@$IP '
                    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
                    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
                    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
                    sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
                    sudo apt-get update
                    sudo apt-get -y install cuda 
                    exit
                    '
                '''
            }
}

def updateAnsibleConfig() {
    script {
        dir ('ansible-config'){
            echo '> Updating Ansible hosts config ...'
            sh label: '', script: """
            sudo sed -i 's@<username>@''ansible-instance-2''@g' hosts.yml
            sudo sed -i -e \'s/<host>/${IP}/g\' hosts.yml
            """
            echo '> Updating Ansible playbook ...'
            sh label: '', script: """
            sudo sed -i 's@<username>@''ansible-instance-2''@g' site.yml
            sudo sed -i -e \'s/<host>/${IP}/g\' site.yml
            """
            echo '> Updating Ansible variable list ...'
            sh label: '', script: """
            sudo sed -i -e \'s/<usecase_id>/${USECASE}/g\' varlist.yml
            sudo sed -i -e \'s/<application_id>/${APPLICATION_ID}/g\' varlist.yml
            sudo sed -i -e \'s/<script_name>/${SCRIPT_NAME}/g\' varlist.yml
            """
            if (env.SCRIPT_NAME == "classifier.py") { 
                sh "sudo sed -i 's@<is_classifier>@''true''@g' varlist.yml"
            } else { 
                sh "sudo sed -i 's@<is_classifier>@''false''@g' varlist.yml"
            }
        }
    }
}

def runAnsiblePlaybook() {
    echo '> Deploying the application ...'
    withCredentials([file(credentialsId: 'ansible-vault', variable: 'VAULT_PASSWORD')]) {
    sh label: '', script: '''
        sudo ansible-playbook ansible-config/site.yml -T 150 -i ansible-config/hosts.yml --private-key ${SSH_KEY_FILE} --vault-password-file $VAULT_PASSWORD -u ansible-instance-2 -e 'ansible_python_interpreter=/usr/bin/python3'
        '''
    }
}

def downloadAnsibleConfig() {
    sh "sudo rm -rf ansible-config"
    sh "mkdir -p ansible-config"
    dir ('ansible-config'){
        echo '> Downloading Ansible Config Files ...'
        withAWS(credentials: 'aws-devops', region: 'us-east-2') {
            sh '''
            sudo rm -rf *
            sudo aws s3 sync s3://model-build-ansible-config/ ./
            '''
        }
    }
}

def checkOpenPortSSH() {
    sh label: '', script: '''
    ssh_status=0
    elapsed=0
    sleep_time=5
    timeout=60
    until [ $ssh_status -eq 1 ];
    do
        echo "$ssh_status"
        sleep $sleep_time
        ssh=(`nmap -p 22 --open -sV ${IP} -Pn | awk '/open/{print $2}'`)
        if [[ $ssh == "open" ]]
        then
            ssh_status=1
        fi
        ((elapsed=elapsed+sleep_time))
        if [[ $elapsed -eq $timeout ]]
        then
            exit 1
        fi
    done
    '''
}

def terminateInstance() {
    if (env.CLOUD_PROVIDER == "aws") {
        sh "aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
    } else if (env.CLOUD_PROVIDER == "gcp") {
        sh "gcloud compute instances delete $INSTANCE_NAME --zone ${REGION}-${GCP_ZONE} --delete-disks all"
    }
} 

def notifyPipelineBeginning () {
    try {
        def pipelineStartedPayLoad = """
        {"usecase_id":"$USECASE","model_id":1,"completed": false,"step":"build","success":true,"msg":"Started build job","type": "build"}
        """
        httpRequest httpMode: 'POST', timeout: 20, customHeaders: [[name: 'Content-Type', value: 'application/json'],[name: 'Verification', value: 'E1D1567C1832387FAC911C2FADB9D-$']], requestBody: pipelineStartedPayLoad, url: env.POST_BUILD_URL
    }
    catch (Exception ex) {
        echo 'downstream service unavailable'
    }
}

def notifyPipelineSuccess () {
    try {
        def pipelineSuccessPayLoad = """
        {"usecase_id":"$USECASE","model_id":1,"completed": true,"step":"build","success":true,"msg":"Started build job","type": "build"}
        """
        httpRequest httpMode: 'POST', timeout: 20, customHeaders: [[name: 'Content-Type', value: 'application/json'],[name: 'Verification', value: 'E1D1567C1832387FAC911C2FADB9D-$']], requestBody: pipelineSuccessPayLoad, url: env.POST_BUILD_URL
    }
    catch (Exception ex) {
        echo 'downstream service unavailable'
    }
}

def notifyPipelineFailure () {
    try {
        def pipelineFailurePayLoad = """
        {"usecase_id":"$USECASE","model_id":1,"completed": true,"step":"build","success":false,"msg":"Started build job","type": "build"}
        """
        httpRequest httpMode: 'POST', timeout: 20, customHeaders: [[name: 'Content-Type', value: 'application/json'],[name: 'Verification', value: 'E1D1567C1832387FAC911C2FADB9D-$']], requestBody: pipelineFailurePayLoad, url: env.POST_BUILD_URL
    }
    catch (Exception ex) {
        echo 'downstream service unavailable'
    }
}

def notifyBuildBeginStatus () {
    try {
        def stageBeginsPayLoad = """
        {"usecase_id":"$USECASE","model_id":1,"completed": false,"step":"$MODEL_NAME","success":true,"msg":"Started build","type": "build"}
        """
        httpRequest httpMode: 'POST', customHeaders: [[name: 'Content-Type', value: 'application/json'],[name: 'Verification', value: 'E1D1567C1832387FAC911C2FADB9D-$']], requestBody: stageBeginsPayLoad, url: env.POST_BUILD_URL
    }
    catch (Exception ex) {
        echo 'downstream service unavailable'
    }
}

def notifyBuildSuccessStatus () {
    try {
        def stageCompletePayLoad = """
        {"usecase_id":"$USECASE","model_id":1,"completed": true,"step":"$MODEL_NAME","success":true,"msg":"Build successfull","type": "build"}
        """
        httpRequest httpMode: 'POST', customHeaders: [[name: 'Content-Type', value: 'application/json'],[name: 'Verification', value: 'E1D1567C1832387FAC911C2FADB9D-$']], requestBody: stageCompletePayLoad, url: env.POST_BUILD_URL
    }
    catch (Exception ex) {
        echo 'downstream service unavailable'
    }
}

def notifyBuildFailureStatus () {
    try {
        def stageFailurePayLoad = """
        {"usecase_id":"$USECASE","model_id":1,"completed": true,"step":"$MODEL_NAME","success":false,"msg":"Build failed","type": "build"}
        """
        httpRequest httpMode: 'POST', customHeaders: [[name: 'Content-Type', value: 'application/json'],[name: 'Verification', value: 'E1D1567C1832387FAC911C2FADB9D-$']], requestBody: stageFailurePayLoad, url: env.POST_BUILD_URL
    }
    catch (Exception ex) {
        echo 'downstream service unavailable'
    }
}

def getModelName() {
    def lastCharacters = env.SCRIPT_NAME.substring(env.SCRIPT_NAME.length() - 4,env.SCRIPT_NAME.length() - 3)
    echo "${lastCharacters}"
    echo "^[0-9]*\$"
    if (lastCharacters.matches("^[0-9]*\$")) {
        env.MODEL_NAME="Model"+lastCharacters
    } else {
        env.MODEL_NAME="classifier"
    }
    echo "$MODEL_NAME"
}
