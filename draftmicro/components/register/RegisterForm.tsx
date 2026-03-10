import React from "react"

const LoginForm: React.FC = () => {
  return (
    <>
      <div className="form-group">
        <label>Username or Email address</label>
        <input type="text" />
      </div>

      <div className="form-group">
        <label>Password</label>
        <input type="password" />
      </div>

      <div className="form-group">
        <label>Confirm Password</label>
        <input type="password" />
      </div>

      <div className="options">
        <label>
          <input type="checkbox" /> Remember me
        </label>

        <div>
          <button className="confirmed-btn">ConFirme</button>
        </div>
      </div>
    </>
  )
}

export default LoginForm