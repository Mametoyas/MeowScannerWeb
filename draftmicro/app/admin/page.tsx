"use client"
import { useState } from "react";
import "../../styles/Admin.css";
import AdminNavbar from "@/components/admin/AdminNavbar";
import AdminCatList from "@/components/admin/AdminCatList";
import CatModal from "@/components/admin/CatModal";
import { Cat } from "@/components/search/CatList";

export default function AdminPage() {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [showModal, setShowModal] = useState(false);
  const [editingCat, setEditingCat] = useState<Cat | null>(null);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };

  const handleAddCat = () => {
    setEditingCat(null);
    setShowModal(true);
  };

  const handleEditCat = (cat: Cat) => {
    setEditingCat(cat);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setEditingCat(null);
  };

  return (
    <>
      <AdminNavbar 
        searchTerm={searchTerm} 
        onSearchChange={handleSearchChange}
        onAddCat={handleAddCat}
      />

      <div className="admin-container">
        <AdminCatList 
          searchTerm={searchTerm} 
          onEditCat={handleEditCat}
        />
      </div>

      {showModal && (
        <CatModal 
          cat={editingCat}
          onClose={handleCloseModal}
        />
      )}
    </>
  );
}