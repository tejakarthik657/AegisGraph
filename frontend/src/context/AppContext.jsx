import { createContext, useContext, useMemo, useState } from "react";

const AppContext = createContext(null);
export function AppProvider({ children }) { const [sidebarOpen, setSidebarOpen] = useState(true); const value = useMemo(() => ({ sidebarOpen, setSidebarOpen }), [sidebarOpen]); return <AppContext.Provider value={value}>{children}</AppContext.Provider>; }
export const useAppContext = () => useContext(AppContext);
export default AppContext;
