# CAP-STOP-001 修复前冻结期望

所有动态输入和transport均为SYNTHETIC；使用真实pinned Counter、Provider构造函数和新Ledger，只向本run的新fixture目录写入。禁止网络、LiveHTTP、execute_authorized和旧输出目录写入。修前与修后使用同一脚本和相同期望，分别留存不可覆盖的输出。

生成超cap应0次transport；初评超cap应只完成1次生成；第三超cap应只完成生成+A+B的3次。任一情况均应ledger.broken=true，未调用评分为rating=null，之后任何槽位不得再调用transport。预留不退款，未预留槽位不伪造收费。

同样检查context超限、计数literal、业务绑定、配置漂移及预算拒绝；已获usage/身份异常保持停批和raw。正常13个SYNTHETIC单元52次应成功，固定cap总预留1.2460032；普通输出截断/评分格式失败保留既有规则，不扩大为全部失败停批。
