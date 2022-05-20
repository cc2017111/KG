#########################################################################
# File Name: wechaty_test.sh
# Author: jyh
# mail: 2832337912@qq.com
# Created Time: 2022年05月17日 星期二 09时04分45秒
#########################################################################
#!/bin/bash
export WECHATY_LOG="verbose"
export WECHATY_PUPPET="wechaty-puppet-padlocal"
export WECHATY_PUPPET_PADLOCAL_TOKEN="puppet_padlocal_ff92c21f7c344578b013290919cff8a4"

export WECHATY_PUPPET_SERVER_PORT="9099"
export WECHATY_TOKEN="12345"

docker run -ti \
  --name wechaty_puppet_service_token_gateway \
  --rm \
  -e WECHATY_LOG \
  -e WECHATY_PUPPET \
  -e WECHATY_PUPPET_PADLOCAL_TOKEN \
  -e WECHATY_PUPPET_SERVER_PORT \
  -e WECHATY_TOKEN \
  -p "$WECHATY_PUPPET_SERVER_PORT:$WECHATY_PUPPET_SERVER_PORT" \
  wechaty/wechaty:0.56
