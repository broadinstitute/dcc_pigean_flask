pipeline {
    options {
        timestamps()
        skipDefaultCheckout()
        disableConcurrentBuilds()
    }
    agent {
        node { label 'translator && aws && build' }
    }
    parameters {
        string(name: 'BUILD_VERSION', defaultValue: '', description: 'This build version to deploy (optional)')
        string(name: 'AWS_REGION', defaultValue: 'us-east-1', description: 'AWS Region to deploy')
    }
    triggers {
        pollSCM('H/5 * * * *')
    }
    environment {
        IMAGE_NAME = "853771734544.dkr.ecr.us-east-1.amazonaws.com/gene-nmf"
        KUBERNETES_BLUE_CLUSTER_NAME = "translator-eks-ci-blue-cluster"
        DEPLOY_ENV = "ci"
    }
    stages {
        stage('Checkout source code') {
            steps {
                cleanWs()
                checkout scm
            }
        }
        stage('Build Version'){
            when {
                allOf {
                    expression {
                        return !params.BUILD_VERSION
                    }
                }
            }
            steps{
                script {
                    BUILD_VERSION_GENERATED = VersionNumber(
                        versionNumberString: 'v${BUILD_YEAR, XX}.${BUILD_MONTH, XX}${BUILD_DAY, XX}.${BUILDS_TODAY}',
                        projectStartDate:    '1970-01-01',
                        skipFailedBuilds:    true)
                    currentBuild.displayName = BUILD_VERSION_GENERATED
                    env.BUILD_VERSION = BUILD_VERSION_GENERATED
                    env.BUILD = 'true'
                }
            }
        }
        stage('Build Docker') {
           when { expression { return env.BUILD == 'true' }}
            steps {
                script {
                    docker.build(env.IMAGE_NAME, "--no-cache .")
                    sh '''
                    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin  853771734544.dkr.ecr.us-east-1.amazonaws.com
                    '''
                    docker.image(env.IMAGE_NAME).push("${BUILD_VERSION}")
                }
            }
        }
        stage('Deploy to AWS EKS Blue') {
            agent {
                label 'translator && ci && deploy'
            }
            steps {
                configFileProvider([
                    configFile(fileId: 'values-ci.yaml', targetLocation: 'values-ncats.yaml'),
                    configFile(fileId: 'prepare.sh', targetLocation: 'prepare.sh')
                ]){
                    script {
                        sh '''
                        aws --region ${AWS_REGION} eks update-kubeconfig --name ${KUBERNETES_BLUE_CLUSTER_NAME}
                        /bin/bash prepare.sh
                        cd translator-ops/ops/geneticskp/gene-nmf/
                        /bin/bash deploy.sh
                        '''
                    }
                }
            }
            post {
                always {
                    echo " Clean up the workspace in deploy node!"
                    cleanWs()
                }
            }
        }
    }
}