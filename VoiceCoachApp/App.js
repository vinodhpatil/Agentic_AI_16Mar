import React from 'react';
import { registerRootComponent } from 'expo';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import CoachScreen from './src/screens/CoachScreen';

function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <CoachScreen />
    </SafeAreaProvider>
  );
}

registerRootComponent(App);

export default App;
