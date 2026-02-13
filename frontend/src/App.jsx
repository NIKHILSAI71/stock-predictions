import { StockProvider, useStock } from './context/StockContext';
import Navbar from './components/layout/Navbar';
import Footer from './components/layout/Footer';
import LoadingOverlay from './components/layout/LoadingOverlay';
import StockHeader from './components/stock/StockHeader';
import InsightPanel from './components/ai/InsightPanel';
import DataGrid from './components/stock/DataGrid';
import PriceTargets from './components/stock/PriceTargets';
import HistoricalValidation from './components/stock/HistoricalValidation';
import TransparencyPanel from './components/stock/TransparencyPanel';
import DataQualityPanel from './components/stock/DataQualityPanel';
import ErrorPage from './components/pages/ErrorPage';

import useStockData from './hooks/useStockData';

// Main Content Component to access Context
const MainContent = () => {
  const { symbol, stockData, error } = useStock();
  useStockData(); // Activate data fetching logic

  if (error) {
    return <ErrorPage error={error} />;
  }

  return (
    <div className="app-container">
      <Navbar />

      <main className="main-content">
        {symbol && stockData ? (
          <>
            <StockHeader />
            <InsightPanel />
            <TransparencyPanel />
            <DataQualityPanel />
            <PriceTargets />
            <HistoricalValidation />
            <DataGrid />
          </>
        ) : null}
      </main>

      <Footer />
      <LoadingOverlay />
    </div>
  );
};

function App() {
  return (
    <StockProvider>
      <MainContent />
    </StockProvider>
  );
}

export default App;
