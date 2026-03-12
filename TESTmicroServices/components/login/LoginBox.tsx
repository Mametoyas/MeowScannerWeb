import React from "react"
import LoginForm from "./LoginForm"

const LoginBox: React.FC = () => {
  return (
    <div className="login-box">
      <h1>MEOW SCANNER</h1>
      <LoginForm />
    </div>
  )
}

export default LoginBox