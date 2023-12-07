LCC_CONTENT=`openssl base64 -A -in install-packages.sh`

aws sagemaker create-studio-lifecycle-config \
--studio-lifecycle-config-name install-pip-package-on-kernel \
--studio-lifecycle-config-content $LCC_CONTENT \
--studio-lifecycle-config-app-type KernelGateway

{
    "StudioLifecycleConfigArn": "arn:aws:sagemaker:us-east-1:034700280673:studio-lifecycle-config/install-pip-package-on-kernel"
}

aws sagemaker create-user-profile --domain-id d-f5xesodmzkxg \
--user-profile-name sti-user \
--user-settings '{
"KernelGatewayAppSettings": {
  "LifecycleConfigArns":
    ["arn:aws:sagemaker:us-east-1:034700280673:studio-lifecycle-config/install-pip-package-on-kernel"]
  }
}'

{
    "UserProfileArn": "arn:aws:sagemaker:us-east-1:034700280673:user-profile/d-f5xesodmzkxg/my-new-user"
}

aws sagemaker update-user-profile --domain-id d-f5xesodmzkxg \
--user-profile-name studio-user \
--user-settings '{
"KernelGatewayAppSettings": {
  "LifecycleConfigArns":
    ["arn:aws:sagemaker:us-east-1:034700280673:studio-lifecycle-config/install-pip-package-on-kernel"]
  }
}'

{
    "UserProfileArn": "arn:aws:sagemaker:us-east-1:034700280673:user-profile/d-f5xesodmzkxg/studio-user"
}