"use client"
import { useState, useEffect } from "react";
import CatCard from "./CatCard";

export interface Cat {
  CatID: string;
  CatName: string;
  CatPersonal: string;
  CatDetails: string;
  Prices?: string;
  ImgURL?: string;
}

interface CatListProps {
  searchTerm: string;
}

export default function CatList({ searchTerm }: CatListProps) {
  const [cats, setCats] = useState<Cat[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchCats();
  }, []);

  const fetchCats = async () => {
    try {
      setLoading(true);
      const { DATABASE_API } = await import('../../config/api').then(m => m.getAPIUrls())
      const response = await fetch(`${DATABASE_API}/get-cats`);
      const data = await response.json();
      
      if (response.ok && data.success) {
        setCats(data.cats || []);
      } else {
        setError('Failed to fetch cat data');
      }
    } catch (error) {
      console.error('Error fetching cats:', error);
      setError('Error connecting to database');
    } finally {
      setLoading(false);
    }
  };

  const filteredCats = cats.filter((cat) => 
    cat.CatName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    cat.CatPersonal.toLowerCase().includes(searchTerm.toLowerCase()) ||
    cat.CatDetails.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="cat-list">
        <div className="loading-message">
          <p>🐱 Loading cat breeds...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="cat-list">
        <div className="error-message">
          <p>❌ {error}</p>
          <button onClick={fetchCats} className="retry-btn">
            🔄 Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="cat-list">
      {searchTerm && (
        <div className="search-results-header">
          <p>🔍 Found {filteredCats.length} result(s) for "{searchTerm}"</p>
        </div>
      )}
      
      {filteredCats.length > 0 ? (
        filteredCats.map((cat) => (
          <CatCard key={cat.CatID} cat={cat} />
        ))
      ) : searchTerm ? (
        <div className="no-results">
          <p>😿 No cats found for "{searchTerm}"</p>
          <p>Try searching for:</p>
          <ul>
            <li>Cat breed names (e.g., "Siamese", "Bengal")</li>
            <li>Personality traits (e.g., "playful", "calm")</li>
            <li>Physical features (e.g., "long hair", "blue eyes")</li>
          </ul>
        </div>
      ) : (
        <div className="all-cats">
          <h2>🐾 All Cat Breeds ({cats.length})</h2>
          {cats.map((cat) => (
            <CatCard key={cat.CatID} cat={cat} />
          ))}
        </div>
      )}
    </div>
  );
}