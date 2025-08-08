#include "base/base.h"
#include <utility>
namespace base{
Status::Status(int code, std::string error_message)
    :code_(code), message_(std::move(error_message)){}

Status& Status::operator=(int code){
    code_ = code;
    return *this;
}

bool Status::operator==(int code) const{
    if(code_ == code) return true;
    else return false;
}

bool Status::operator!=(int code) const{
    if(code_ != code) return true;
    else return false;
}

Status::operator int() const { return code_; }

Status::operator bool() const {
  return code_ == kSuccess;
}

int32_t Status::get_err_code() const {
  return code_;
}

const std::string& Status::get_err_msg() const { return message_; }

void Status::set_err_msg(const std::string& err_msg) { message_ = err_msg; }

namespace error{
Status Success(const std::string& err_msg) { return Status{kSuccess, err_msg}; }

Status FunctionNotImplement(const std::string& err_msg) {
  return Status{kFunctionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) {
  return Status{kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
  return Status{kModelParseError, err_msg};
}

Status InternalError(const std::string& err_msg) {
  return Status{kInternalError, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
  return Status{kInvalidArgument, err_msg};
}

Status KeyHasExits(const std::string& err_msg) {
  return Status{kKeyValueHasExist, err_msg};
}
} // namespace error

} // namespace base