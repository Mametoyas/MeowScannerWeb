export const isUserLoggedIn = (): boolean => {
  if (typeof window === 'undefined') return false;
  
  const sessionUser = sessionStorage.getItem('user');
  const localUser = localStorage.getItem('user');
  
  return !!(sessionUser || localUser);
};

export const getUserData = () => {
  if (typeof window === 'undefined') return null;
  
  const sessionUser = sessionStorage.getItem('user');
  const localUser = localStorage.getItem('user');
  
  const userData = sessionUser || localUser;
  return userData ? JSON.parse(userData) : null;
};

export const logoutUser = () => {
  if (typeof window === 'undefined') return;
  
  sessionStorage.removeItem('user');
  localStorage.removeItem('user');
  window.location.href = '/login';
};