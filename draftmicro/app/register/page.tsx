import "../../styles/Register.css"
import Navbar from "@/components/register/Navbar"
import CatImage from "@/components/register/CatImage"
import RegisterBox from "@/components/register/RegisterBox"

export default function RegisterPage() {
  return (
    <>
      <Navbar />

      <div className="container">
        <CatImage />
        <RegisterBox />
      </div>
    </>
  )
}