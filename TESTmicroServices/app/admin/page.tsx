'use client';

import { useState, useEffect } from 'react';
import { getUserData } from '../../utils/auth';
import ProtectedRoute from '@/components/ProtectedRoute';
import '../../styles/Admin.css';

interface CatData {
  CatID: string;
  CatName: string;
  CatPersonal: string;
  CatDetails: string;
  Prices?: string;
  ImgURL?: string;
}

interface UserData {
  ID: string;
  Name: string;
  Username: string;
  Role: string;
}

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<'cats' | 'users'>('cats');
  const [cats, setCats] = useState<CatData[]>([]);
  const [users, setUsers] = useState<UserData[]>([]);
  const [filteredCats, setFilteredCats] = useState<CatData[]>([]);
  const [filteredUsers, setFilteredUsers] = useState<UserData[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCat, setSelectedCat] = useState<CatData | null>(null);
  const [selectedUser, setSelectedUser] = useState<UserData | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [DATABASE_API, setDATABASE_API] = useState('');

  const [catFormData, setCatFormData] = useState<CatData>({
    CatID: '',
    CatName: '',
    CatPersonal: '',
    CatDetails: '',
    Prices: '',
    ImgURL: ''
  });

  const [userFormData, setUserFormData] = useState<UserData & { Password?: string }>({
    ID: '',
    Name: '',
    Username: '',
    Role: 'user',
    Password: ''
  });

  useEffect(() => {
    const loadAPI = async () => {
      const { DATABASE_API } = await import('../../config/api').then(m => m.getAPIUrls());
      setDATABASE_API(DATABASE_API);
    };
    loadAPI();
  }, []);

  useEffect(() => {
    if (DATABASE_API) {
      if (activeTab === 'cats') {
        fetchCats();
      } else {
        fetchUsers();
      }
    }
  }, [activeTab, DATABASE_API]);

  useEffect(() => {
    if (activeTab === 'cats') {
      const filtered = cats.filter(cat =>
        cat.CatName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        cat.CatID.toLowerCase().includes(searchTerm.toLowerCase()) ||
        cat.CatPersonal.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredCats(filtered);
    } else {
      const filtered = users.filter(user =>
        user.Name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        user.Username.toLowerCase().includes(searchTerm.toLowerCase()) ||
        user.Role.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredUsers(filtered);
    }
  }, [searchTerm, cats, users, activeTab]);

  const fetchUsers = async () => {
    if (!DATABASE_API) return;
    try {
      const response = await fetch(`${DATABASE_API}/get-users`);
      const data = await response.json();
      if (response.ok) {
        const currentUser = getUserData();
        const filteredUsers = data.users?.filter((user: UserData) => user.ID !== currentUser?.user_id) || [];
        setUsers(filteredUsers);
      } else {
        setMessage('Failed to fetch user data');
      }
    } catch (error) {
      console.error('Error fetching users:', error);
      setMessage('Error fetching user data');
    } finally {
      setLoading(false);
    }
  };

  const fetchCats = async () => {
    if (!DATABASE_API) return;
    try {
      const response = await fetch(`${DATABASE_API}/get-cats`);
      const data = await response.json();
      if (response.ok) {
        setCats(data.cats || []);
      } else {
        setMessage('Failed to fetch cat data');
      }
    } catch (error) {
      console.error('Error fetching cats:', error);
      setMessage('Error fetching cat data');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };

  const handleEditCat = (cat: CatData) => {
    setSelectedCat(cat);
    setCatFormData(cat);
    setIsEditing(true);
    setIsAdding(false);
  };

  const handleEditUser = (user: UserData) => {
    setSelectedUser(user);
    setUserFormData({ ...user, Password: '' });
    setIsEditing(true);
    setIsAdding(false);
  };

  const handleAddCat = () => {
    const newId = `C${String(cats.length + 1).padStart(4, '0')}`;
    setCatFormData({
      CatID: newId,
      CatName: '',
      CatPersonal: '',
      CatDetails: '',
      Prices: '',
      ImgURL: ''
    });
    setIsAdding(true);
    setIsEditing(false);
    setSelectedCat(null);
  };

  const handleAddUser = () => {
    const newId = `U${String(users.length + 1).padStart(5, '0')}`;
    setUserFormData({
      ID: newId,
      Name: '',
      Username: '',
      Role: 'user',
      Password: ''
    });
    setIsAdding(true);
    setIsEditing(false);
    setSelectedUser(null);
  };

  const handleCatFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setCatFormData({
      ...catFormData,
      [e.target.name]: e.target.value
    });
  };

  const handleUserFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setUserFormData({
      ...userFormData,
      [e.target.name]: e.target.value
    });
  };

  const handleSaveCat = async () => {
    if (!DATABASE_API) return;
    try {
      const url = isAdding ? `${DATABASE_API}/add-cat` : `${DATABASE_API}/update-cat`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(catFormData),
      });

      const data = await response.json();
      if (response.ok) {
        setMessage(isAdding ? 'Cat added successfully!' : 'Cat updated successfully!');
        fetchCats();
        handleCancel();
      } else {
        setMessage(data.message || 'Operation failed');
      }
    } catch (error) {
      console.error('Error saving cat:', error);
      setMessage('Error saving cat data');
    }
  };

  const handleSaveUser = async () => {
    if (!DATABASE_API) return;
    try {
      const url = isAdding ? `${DATABASE_API}/add-user` : `${DATABASE_API}/update-user`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userFormData),
      });

      const data = await response.json();
      if (response.ok) {
        setMessage(isAdding ? 'User added successfully!' : 'User updated successfully!');
        fetchUsers();
        handleCancel();
      } else {
        setMessage(data.message || 'Operation failed');
      }
    } catch (error) {
      console.error('Error saving user:', error);
      setMessage('Error saving user data');
    }
  };

  const handleDeleteUser = async (userId: string) => {
    if (!DATABASE_API) return;
    if (!confirm('Are you sure you want to delete this user?')) return;

    try {
      const response = await fetch(`${DATABASE_API}/delete-user`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_id: userId }),
      });

      const data = await response.json();
      if (response.ok) {
        setMessage('User deleted successfully!');
        fetchUsers();
      } else {
        setMessage(data.message || 'Delete failed');
      }
    } catch (error) {
      console.error('Error deleting user:', error);
      setMessage('Error deleting user');
    }
  };

  const handleDeleteCat = async (catId: string) => {
    if (!DATABASE_API) return;
    if (!confirm('Are you sure you want to delete this cat?')) return;

    try {
      const response = await fetch(`${DATABASE_API}/delete-cat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ cat_id: catId }),
      });

      const data = await response.json();
      if (response.ok) {
        setMessage('Cat deleted successfully!');
        fetchCats();
      } else {
        setMessage(data.message || 'Delete failed');
      }
    } catch (error) {
      console.error('Error deleting cat:', error);
      setMessage('Error deleting cat');
    }
  };

  const handleCancel = () => {
    setIsEditing(false);
    setIsAdding(false);
    setSelectedCat(null);
    setSelectedUser(null);
    setCatFormData({
      CatID: '',
      CatName: '',
      CatPersonal: '',
      CatDetails: '',
      Prices: '',
      ImgURL: ''
    });
    setUserFormData({
      ID: '',
      Name: '',
      Username: '',
      Role: 'user',
      Password: ''
    });
  };

  return (
    <ProtectedRoute requireAdmin={true}>
      <div className="admin-container">
        <header className="admin-header">
          <h1>Admin Panel - Management</h1>
          <div className="admin-controls">
            <div className="tab-buttons">
              <button 
                className={`tab-btn ${activeTab === 'cats' ? 'active' : ''}`}
                onClick={() => setActiveTab('cats')}
              >
                Cats
              </button>
              <button 
                className={`tab-btn ${activeTab === 'users' ? 'active' : ''}`}
                onClick={() => setActiveTab('users')}
              >
                Users
              </button>
            </div>
            <input
              type="text"
              placeholder={`Search ${activeTab}...`}
              value={searchTerm}
              onChange={handleSearch}
              className="search-input"
            />
            <button 
              onClick={activeTab === 'cats' ? handleAddCat : handleAddUser} 
              className="add-btn"
            >
              + Add New {activeTab === 'cats' ? 'Cat' : 'User'}
            </button>
            <button 
              onClick={() => window.location.href = '/main'} 
              className="home-btn"
            >
              Home
            </button>
          </div>
        </header>

        {message && (
          <div className={`message ${message.includes('successfully') ? 'success' : 'error'}`}>
            {message}
          </div>
        )}

        <div className="admin-content">
          {/* Data List */}
          <div className="data-list">
            <h2>{activeTab === 'cats' ? 'Cat Database' : 'User Database'} ({activeTab === 'cats' ? filteredCats.length : filteredUsers.length})</h2>
            {loading ? (
              <div className="loading">Loading...</div>
            ) : (
              <div className="data-grid">
                {activeTab === 'cats' ? (
                  filteredCats.map((cat) => (
                    <div key={cat.CatID} className="data-card">
                      <div className="data-header">
                        <h3>{cat.CatName}</h3>
                        <span className="data-id">{cat.CatID}</span>
                      </div>
                      <p className="data-info">{cat.CatPersonal.substring(0, 100)}...</p>
                      <div className="data-actions">
                        <button onClick={() => handleEditCat(cat)} className="edit-btn">
                          Edit
                        </button>
                        <button onClick={() => handleDeleteCat(cat.CatID)} className="delete-btn">
                          Delete
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  filteredUsers.map((user) => (
                    <div key={user.ID} className="data-card">
                      <div className="data-header">
                        <h3>{user.Name}</h3>
                        <span className="data-id">{user.ID}</span>
                      </div>
                      <p className="data-info">Username: {user.Username}</p>
                      <p className="data-info">Role: {user.Role}</p>
                      <div className="data-actions">
                        <button onClick={() => handleEditUser(user)} className="edit-btn">
                          Edit
                        </button>
                        <button onClick={() => handleDeleteUser(user.ID)} className="delete-btn">
                          Delete
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          {/* Edit/Add Form */}
          {(isEditing || isAdding) && (
            <div className="form-panel">
              <h2>{isAdding ? `Add New ${activeTab === 'cats' ? 'Cat' : 'User'}` : `Edit ${activeTab === 'cats' ? 'Cat' : 'User'}`}</h2>
              <form onSubmit={(e) => { e.preventDefault(); activeTab === 'cats' ? handleSaveCat() : handleSaveUser(); }}>
                {activeTab === 'cats' ? (
                  <>
                    <div className="form-group">
                      <label>Cat ID</label>
                      <input
                        type="text"
                        name="CatID"
                        value={catFormData.CatID}
                        onChange={handleCatFormChange}
                        disabled={isEditing}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label>Cat Name</label>
                      <input
                        type="text"
                        name="CatName"
                        value={catFormData.CatName}
                        onChange={handleCatFormChange}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label>Cat Personality</label>
                      <textarea
                        name="CatPersonal"
                        value={catFormData.CatPersonal}
                        onChange={handleCatFormChange}
                        rows={4}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label>Cat Details</label>
                      <textarea
                        name="CatDetails"
                        value={catFormData.CatDetails}
                        onChange={handleCatFormChange}
                        rows={6}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label>Prices (THB)</label>
                      <input
                        type="text"
                        name="Prices"
                        value={catFormData.Prices || ''}
                        onChange={handleCatFormChange}
                        placeholder="e.g. 5,000-15,000"
                      />
                    </div>

                    <div className="form-group">
                      <label>Image URL</label>
                      <input
                        type="text"
                        name="ImgURL"
                        value={catFormData.ImgURL || ''}
                        onChange={handleCatFormChange}
                        placeholder="https://..."
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <div className="form-group">
                      <label>User ID</label>
                      <input
                        type="text"
                        name="ID"
                        value={userFormData.ID}
                        onChange={handleUserFormChange}
                        disabled={isEditing}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label>Name</label>
                      <input
                        type="text"
                        name="Name"
                        value={userFormData.Name}
                        onChange={handleUserFormChange}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label>Username</label>
                      <input
                        type="text"
                        name="Username"
                        value={userFormData.Username}
                        onChange={handleUserFormChange}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label>Password {isEditing && '(leave blank to keep current)'}</label>
                      <input
                        type="password"
                        name="Password"
                        value={userFormData.Password}
                        onChange={handleUserFormChange}
                        required={isAdding}
                      />
                    </div>

                    <div className="form-group">
                      <label>Role</label>
                      <select
                        name="Role"
                        value={userFormData.Role}
                        onChange={handleUserFormChange}
                        required
                      >
                        <option value="user">User</option>
                        <option value="admin">Admin</option>
                      </select>
                    </div>
                  </>
                )}

                <div className="form-actions">
                  <button type="submit" className="save-btn">
                    {isAdding ? `Add ${activeTab === 'cats' ? 'Cat' : 'User'}` : 'Save Changes'}
                  </button>
                  <button type="button" onClick={handleCancel} className="cancel-btn">
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          )}
        </div>
      </div>
    </ProtectedRoute>
  );
}