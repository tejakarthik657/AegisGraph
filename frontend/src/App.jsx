import { BrowserRouter } from "react-router-dom";

import AppRouter from "./app/AppRouter";
import Header from "./components/shared/Header";
import Sidebar from "./components/shared/Sidebar";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />

        <div className="main-area">
          <Header />

          <main className="main-content">
            <AppRouter />
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}