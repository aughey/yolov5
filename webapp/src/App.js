import './App.css';
import mqtt from 'mqtt'
import { useState, useEffect, useRef } from 'react';

const host = 'ws://127.0.0.1:9001/mqtt'
const options = {
    keepalive: 60,
}
const client = mqtt.connect(host, options)

client.on('error', (err) => {
    console.log('Connection error: ', err)
    client.end()
})

client.on('reconnect', () => {
    console.log('Reconnecting...')
})
client.on('connect', () => {
    console.log('Client connected:')
    // Subscribe
    client.subscribe('yolov5' , function (err) {
    });
})
//client.on('message', (message,data) => {
//    data = JSON.parse(data)
//    console.log(data)
//})

function useMqtt(topic) {
    const [data, setData] = useState(null)

    useEffect(() => {
        var sub = (a, b) => {
            if (a === topic) {
                setData(b)
            }
        }
        client.on('message', sub);
        return () => {
            client.removeListener('message', sub);
        }
    },[topic])

    return data
}

function Once() {
    var nextdata = null;
    var ready = true;

    var me = {
        setData: (d) => {
            if (ready) {
                ready = false;
                me.nextData(d);
            } else {
                nextdata = d;
            }
        },
        ready: () => {
            if (nextdata) {
                ready = false;
                var n = nextdata
                nextdata = null
                me.nextData(n)
            } else {
                ready = true;
            }
        }
    }
    return me;
}

function Overlay() {
    var yolo = useMqtt('yolov5')
    var [url, setUrl] = useState()

    var once = useRef()
    if (!once.current) {
        once.current = Once()
        once.current.nextData = d => setUrl('http://localhost:5000/get/' + d.uuid + ".jpg")
    }

    useEffect(() => {
        if (!yolo) {
            return;
        }
        var data = JSON.parse(yolo);
        once.current.setData(data)
    }, [yolo])

  

    if (url) {
        return (<img alt='frame' onLoad={() => once.current.ready()} src={url} />)
    } else {
        return null;
    }
}

function App() {
    

  
  return (
    <div className="App">
      <header className="App-header">
              <Overlay/>
              <p>
                  
                  Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
